"""Opt-in bounded archive retrieval for historical real-price shock discovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.historical_gdelt_scout import scout_historical_gdelt


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover archived NVIDIA media references")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.run and args.output is None:
            raise ValueError("Real GDELT retrieval requires private --output")
        if args.output is not None and (
            args.output.is_symlink() or args.output.exists()
            or not args.output.parent.is_dir() or args.output.parent.is_symlink()
        ):
            raise ValueError("Private output must be new within an existing directory")
        result = scout_historical_gdelt(
            study_path=args.study, rank=args.rank, run=args.run,
        )
        if args.run:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(args.output, flags, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            except BaseException:
                args.output.unlink(missing_ok=True)
                raise
        print(json.dumps({
            "status": result["status"], "shockRank": result["shockRank"],
            "shockDate": result["shockDate"],
            "archivesPlanned": len(result["archivePlan"]),
            "archivesFetched": result["archivesFetched"],
            "matchedReferences": sum(x["matchingDocumentsTotal"]
                                     for x in result["archives"]),
            "previewSlots": result["archivePlan"] if args.preview else [],
            "outputCreated": args.run, "trainingReady": False,
        }, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"historical-gdelt-scout: {type(exc).__name__}\n")


if __name__ == "__main__":
    main()
