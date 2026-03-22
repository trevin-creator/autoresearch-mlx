#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/.gen-venv/bin/python"

if [ ! -x "$PY" ]; then
  cat <<'EOF'
Missing .gen-venv Python at ./.gen-venv/bin/python

Bootstrap the generation environment first:
  uv venv --python /usr/local/bin/python3.12 .gen-venv
  uv pip install --python .gen-venv/bin/python numpy genesis-world torch
EOF
  exit 1
fi

cd "$ROOT_DIR"

"$PY" generate_stereo_events.py \
  --duration 0.05 \
  --dt 0.01 \
  --depth-every 1 \
  --out-dir ./_smoke_stereo_events_full \
  "$@"