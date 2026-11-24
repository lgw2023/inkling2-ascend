#!/usr/bin/env bash
# The caller must select a5-950pr or a5-950dt; there is no interchangeable A5 build.
set -euo pipefail
TASK_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${TASK_SCRIPT_DIR}/check_env.py" launch "$@"
