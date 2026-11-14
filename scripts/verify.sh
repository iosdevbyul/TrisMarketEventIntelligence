#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m tris_market_event_intelligence.verification_cli "$@"
