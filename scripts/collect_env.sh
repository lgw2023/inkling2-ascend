#!/usr/bin/env bash
# Read-only, allowlisted report. Does not emit raw npu-smi or environment values.
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${TASK_SCRIPT_DIR}/check_env.py" collect "$@"
