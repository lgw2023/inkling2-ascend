#!/usr/bin/env bash
# Default is dry-run. An implemented and qualified adapter is required to execute.
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${TASK_SCRIPT_DIR}/check_env.py" launch --profile a2 "$@"
