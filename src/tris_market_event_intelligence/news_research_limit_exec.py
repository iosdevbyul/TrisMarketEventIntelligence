"""Set OS CPU resource bounds *before exec*, without unsafe Popen preexec_fn."""
from __future__ import annotations

import os
import resource
import sys


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 4 or args[0] != "--cpu-seconds" or args[2] != "--":
        raise SystemExit("Usage: limit-exec --cpu-seconds N -- ABSOLUTE_EXEC ARG ...")
    try:
        seconds = int(args[1])
    except ValueError as exc:
        raise SystemExit("Invalid CPU seconds") from exc
    if not 1 <= seconds <= 86400:
        raise SystemExit("Invalid CPU seconds")
    command = args[3:]
    if not os.path.isabs(command[0]):
        raise SystemExit("Absolute research executable required")
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
