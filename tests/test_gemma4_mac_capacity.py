"""Conservative Gemma 4 Mac readiness: parse real metrics, never download models."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.gemma4_mac_capacity import (
    _run,
    _vm_memory,
    inspect_gemma4_mac,
)


def command(args: list[str]) -> str:
    if args[-1] == "hw.memsize":
        return str(16 * 1024**3) + "\n"
    if args[-1] == "hw.ncpu":
        return "10\n"
    assert args == ["/usr/bin/vm_stat"]
    return (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                             250000.\n"
        "Pages active:                           200000.\n"
        "Pages inactive:                         250000.\n"
        "Pages speculative:                       50000.\n"
        "Pages wired down:                       200000.\n"
    )


def ollama_handler(
    request: httpx.Request, *, missing: bool = False,
) -> httpx.Response:
    assert request.method == "GET"
    assert request.url.host == "127.0.0.1"
    assert request.url.port == 11434
    if request.url.path == "/api/tags":
        rows = [{"name": "gemma2:latest", "size": 2100000000}]
        if not missing:
            rows.append({"name": "gemma4:e4b", "size": 4500000000})
        return httpx.Response(200, json={"models": rows})
    assert request.url.path == "/api/ps"
    return httpx.Response(200, json={"models": [{"name": "gemma2:latest"}]})


def observe(tmp_path: Path, *, missing: bool = False, runner=command):
    visited: list[str] = []

    def handler(request):
        visited.append(request.url.path)
        return ollama_handler(request, missing=missing)

    with httpx.Client(transport=httpx.MockTransport(handler), trust_env=False) as client:
        report = inspect_gemma4_mac(
            models=("gemma2:latest", "gemma4:e4b"), run=runner,
            client=client, system="Darwin", machine="arm64", data_path=tmp_path,
        )
    return report, visited


def test_readonly_snapshot_does_not_assert_model_fit(tmp_path: Path):
    result, visited = observe(tmp_path)
    assert visited == ["/api/tags", "/api/ps"]
    assert result["status"] == "CAPACITY_NOT_VERIFIED"
    assert result["physicalMemoryGiB"] == 16.0
    assert result["logicalCpuCount"] == 10
    assert result["isAppleSiliconReported"] is True
    assert result["memorySnapshot"]["pageSizeBytes"] == 16384
    assert result["memorySnapshot"]["isGuaranteedFreeForInference"] is False
    assert result["ollama"]["installed"] == {
        "gemma2:latest": True, "gemma4:e4b": True,
    }
    assert result["ollama"]["downloadBytesByModel"]["gemma4:e4b"] == 4500000000
    assert result["ollama"]["modelDownloadSizeIsResidentMemory"] is False
    assert result["realInferenceMemoryFitVerified"] is False
    assert result["newsModelQualityVerified"] is False
    assert result["stockPricesUsed"] is False
    assert result["modelDownloadAttempted"] is False


def test_missing_installed_model_is_not_runtime_ready(tmp_path: Path):
    result, _ = observe(tmp_path, missing=True)
    assert result["status"] == "LOCAL_MODEL_NOT_INSTALLED"
    assert result["ollama"]["installed"]["gemma4:e4b"] is False
    assert result["installedModelsSufficientForRuntime"] is False


def test_low_headroom_blocks_even_if_model_is_installed(tmp_path: Path):
    def low_memory(args: list[str]) -> str:
        if args[-1] != "hw.memsize" and args[-1] != "hw.ncpu":
            return (
                "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free: 5000.\nPages inactive: 10000.\nPages speculative: 1000.\n"
            )
        return command(args)

    result, _ = observe(tmp_path, runner=low_memory)
    assert result["status"] == "MEMORY_HEADROOM_LOW_FOR_SHARED_MAC"


def test_unavailable_ollama_is_not_automatically_started(tmp_path: Path):
    def broken(request):
        raise httpx.ConnectError("service unavailable")

    with httpx.Client(transport=httpx.MockTransport(broken)) as client:
        result = inspect_gemma4_mac(
            models=("gemma2:latest", "gemma4:e4b"), run=command,
            client=client, system="Darwin", machine="x86_64", data_path=tmp_path,
        )
    assert result["status"] == "LOCAL_OLLAMA_UNAVAILABLE"
    assert result["isAppleSiliconReported"] is False
    assert result["ollama"] is None


def test_non_mac_returns_unknown_without_subprocess_or_ollama(tmp_path: Path):
    def never_called(args):
        raise AssertionError("hardware commands must not run")

    with httpx.Client(transport=httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("no requests"))
    )) as client:
        result = inspect_gemma4_mac(
            models=("gemma2:latest", "gemma4:e4b"),
            run=never_called, client=client, system="Linux", data_path=tmp_path,
        )
    assert result["status"] == "UNSUPPORTED_HOST_NOT_MACOS"
    assert result["inferenceMemoryFitVerified"] is False


@pytest.mark.parametrize("bad_vm", [
    "No page size\nPages free: 100.\n",
    "Mach Virtual Memory Statistics: (page size of 100 bytes)\n"
    "Pages free: 10.\nPages inactive: 50.\nPages speculative: 10.\n",
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    "Pages free: 9999999999999.\nPages inactive: 5.\nPages speculative: 5.\n",
])
def test_unreliable_vm_stat_fails_closed(bad_vm: str):
    with pytest.raises(ValueError):
        _vm_memory(bad_vm, 16 * 1024**3)


def test_disallow_arbitrary_sysctl_or_shell_command():
    with pytest.raises(ValueError, match="Unsupported"):
        _run(["/usr/sbin/sysctl", "-n", "kern.hostname"])
    with pytest.raises(ValueError, match="Unsupported"):
        _run(["/bin/sh", "-c", "echo bad"])


def test_invalid_gemma_tag_rejected_before_probing(tmp_path: Path):
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(AssertionError("no probes"))
    )) as client:
        with pytest.raises(ValueError, match="explicitly sized"):
            inspect_gemma4_mac(
                models=("gemma2:latest", "gemma4:latest"),
                run=command, client=client, system="Darwin", data_path=tmp_path,
            )
