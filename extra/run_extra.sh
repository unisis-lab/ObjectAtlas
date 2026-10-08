#!/usr/bin/env bash
set -euo pipefail
extra_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="$extra_dir/../.venv/bin/python"
if [[ ! -x "$python" ]]; then python="$(command -v python3 || command -v python)"; fi
exec "$python" "$extra_dir/run_extra.py" "$@"
