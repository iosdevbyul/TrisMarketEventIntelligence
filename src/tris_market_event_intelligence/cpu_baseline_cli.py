"""Offline, read-only Intel Mac CPU baseline research report command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.cpu_baseline import evaluate_empirical_baseline


def evaluate_report(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        source = json.load(handle)
    if not isinstance(source, dict):
        raise TypeError("Expected a JSON object")
    dataset = source.get("walkforward_dataset", source)
    if not isinstance(dataset, dict):
        raise TypeError("Missing walk-forward dataset")
    return evaluate_empirical_baseline(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline CPU historical mean versus zero-return baseline",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = evaluate_report(args.dataset)
        if args.output is None:
            print(json.dumps(result, ensure_ascii=False))
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(json.dumps({
                "status": "ok", "report": str(args.output),
                "groups": len(result["groups"]),
                "prediction_model_trained": False,
            }))
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        parser.exit(2, f"cpu-baseline: {exc}\n")


if __name__ == "__main__":
    main()
