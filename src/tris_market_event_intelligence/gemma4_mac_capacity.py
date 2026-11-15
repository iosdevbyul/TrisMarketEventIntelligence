"""Read-only macOS Gemma 4 feasibility snapshot; never launches or pulls a model.

Memory pressure is observed at one instant. Free pages are NOT a guaranteed
inference budget, and downloaded model bytes are NOT an RSS estimate.
"""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import httpx

from tris_market_event_intelligence.gemma4_news_comparison import _validate_models

MAX_COMMAND_SECONDS = 5
MAX_OUTPUT = 32768
PAGE_RE = re.compile(r"\bpage size of (\d+) bytes\b")
STAT_RE = re.compile(r"^Pages (free|inactive|speculative|purgeable|wired down|active):\s*(\d+)\.$")
RAM_RESERVE_GIB = 4


def _run(args: list[str]) -> str:
    """Allowlisted read-only macOS utilities, no shell or app modification."""
    if not args or args[0] not in {"/usr/sbin/sysctl", "/usr/bin/vm_stat"}:
        raise ValueError("Unsupported hardware inspection command")
    if args[0] == "/usr/sbin/sysctl" and args[1:] not in (
        ["-n", "hw.memsize"], ["-n", "hw.ncpu"],
    ):
        raise ValueError("Unsupported sysctl key")
    if args[0] == "/usr/bin/vm_stat" and len(args) != 1:
        raise ValueError("Unexpected vm_stat arguments")
    response = subprocess.run(
        args, capture_output=True, text=True,
        timeout=MAX_COMMAND_SECONDS, check=False,
    )
    if response.returncode != 0 or len(response.stdout) > MAX_OUTPUT:
        raise ValueError("Hardware inspection failed or returned oversized data")
    return response.stdout


def _bytes_and_count(run: Callable[[list[str]], str]) -> tuple[int, int]:
    ram = int(run(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip())
    cpus = int(run(["/usr/sbin/sysctl", "-n", "hw.ncpu"]).strip())
    if not 512 * 1024**2 <= ram <= 1024**5 or not 1 <= cpus <= 1024:
        raise ValueError("Hardware capacity outside supported diagnostic range")
    return ram, cpus


def _vm_memory(text: str, total_ram: int) -> dict[str, object]:
    if len(text) > MAX_OUTPUT:
        raise ValueError("Oversized vm_stat result")
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    pages: dict[str, int] = {}
    size = None
    for line in rows:
        found = PAGE_RE.search(line)
        if found:
            size = int(found.group(1))
            continue
        match = STAT_RE.search(line)
        if match:
            pages[match.group(1)] = int(match.group(2))
    if size is None or not 4096 <= size <= 65536 or (
        size & (size - 1)
    ) or not {"free", "inactive", "speculative"} <= pages.keys():
        raise ValueError("vm_stat page capacity cannot be safely interpreted")
    # The sum is a rough snapshot only; inactive may be reclaimed or re-used,
    # and free memory can shrink before inference begins.
    available = sum(pages[name] for name in ("free", "inactive", "speculative")) * size
    if available > total_ram * 2:
        raise ValueError("vm_stat exceeds physical memory plausibility")
    return {
        "pageSizeBytes": size,
        "observedFreeInactiveSpeculativeBytes": available,
        "observedFreeInactiveSpeculativeGiB": round(available / 1024**3, 3),
        "isGuaranteedFreeForInference": False,
        "macOsMemoryPressureVerified": False,
    }


def _local_ollama(models: tuple[str, str], client: httpx.Client) -> dict[str, object]:
    def load(endpoint: str) -> dict[str, object]:
        response = client.get(f"http://127.0.0.1:11434{endpoint}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise TypeError("Unexpected local Ollama model registry")
        return payload

    tags = load("/api/tags")
    running = load("/api/ps")
    installed = {}
    for item in tags["models"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        if item["name"] not in models:
            continue
        size = item.get("size")
        installed[item["name"]] = size if type(size) is int and size >= 0 else None
    active = []
    for item in running["models"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        active.append(item["name"])
    return {
        "reachable": True,
        "installed": {name: name in installed for name in models},
        "downloadBytesByModel": {name: installed.get(name) for name in models},
        "currentlyLoadedModelNames": sorted(set(active)),
        "modelDownloadSizeIsResidentMemory": False,
        "modelDownloadedOrInstalled": False,
    }


def inspect_gemma4_mac(
    *, models: tuple[str, str],
    run: Callable[[list[str]], str] = _run,
    client: httpx.Client,
    system: str | None = None,
    machine: str | None = None,
    data_path: Path | None = None,
) -> dict[str, object]:
    """Return a conservative model-readiness decision based on local facts.

    No readiness success is possible from a static memory snapshot alone:
    only a bounded real inference experiment can establish runtime fit.
    """
    _validate_models(models)
    system = system if system is not None else platform.system()
    machine = machine if machine is not None else platform.machine()
    if system != "Darwin":
        return {
            "schemaVersion": "gemma4_mac_capacity_preflight_v1",
            "status": "UNSUPPORTED_HOST_NOT_MACOS",
            "models": list(models),
            "macOsObserved": False,
            "modelDownloadAttempted": False,
            "productionModelChanged": False,
            "inferenceMemoryFitVerified": False,
            "newsModelQualityVerified": False,
        }
    physical, cpu_count = _bytes_and_count(run)
    vm = _vm_memory(run(["/usr/bin/vm_stat"]), physical)
    disk_path = data_path if data_path is not None else Path.home()
    free_disk = shutil.disk_usage(disk_path).free
    ollama = None
    ollama_problem = None
    try:
        ollama = _local_ollama(models, client)
    except (httpx.HTTPError, TypeError, ValueError):
        ollama_problem = "LOCAL_OLLAMA_UNREACHABLE_OR_UNEXPECTED_RESPONSE"
    available = vm["observedFreeInactiveSpeculativeBytes"]
    reserve_bytes = RAM_RESERVE_GIB * 1024**3
    status = "CAPACITY_NOT_VERIFIED"
    if physical <= reserve_bytes or available <= reserve_bytes:
        status = "MEMORY_HEADROOM_LOW_FOR_SHARED_MAC"
    elif ollama is None:
        status = "LOCAL_OLLAMA_UNAVAILABLE"
    elif not all(ollama["installed"].values()):
        status = "LOCAL_MODEL_NOT_INSTALLED"
    return {
        "schemaVersion": "gemma4_mac_capacity_preflight_v1",
        "status": status,
        "models": list(models),
        "macOsObserved": True,
        "cpuArchitecture": machine,
        "isAppleSiliconReported": machine == "arm64",
        "logicalCpuCount": cpu_count,
        "physicalMemoryBytes": physical,
        "physicalMemoryGiB": round(physical / 1024**3, 3),
        "sharedMacReserveGiB": RAM_RESERVE_GIB,
        "memorySnapshot": vm,
        "diskFreeBytes": free_disk,
        "ollama": ollama,
        "ollamaErrorCode": ollama_problem,
        "installedModelsSufficientForRuntime": False,
        "realInferenceMemoryFitVerified": False,
        "runtimeLatencyVerified": False,
        "newsModelQualityVerified": False,
        "modelDownloadAttempted": False,
        "productionModelChanged": False,
        "productionDatabaseModified": False,
        "stockPricesUsed": False,
        "recommendedNextAction": (
            "INSPECT_REAL_MAC_LOAD_AND_RUN_EXPLICIT_SMALL_BOUNDED_PILOT"
            if status == "CAPACITY_NOT_VERIFIED" else
            "REVIEW_BLOCKING_LOCAL_FINDINGS_WITHOUT_AUTO_INSTALL"
        ),
    }
