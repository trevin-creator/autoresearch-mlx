#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

RUNTIME_VENV="$ROOT_DIR/.venv-checks"
RUNTIME_LOG_DIR="${TMPDIR:-/tmp}/autoresearch-mlx"
mkdir -p "$RUNTIME_LOG_DIR"

REQUIRED_MODULES=(mlx numpy pyarrow regex requests rustbpe tiktoken)

find_python() {
  local candidates=()
  if [ -n "${AUTORESEARCH_PYTHON:-}" ]; then
    candidates+=("${AUTORESEARCH_PYTHON}")
  fi
  candidates+=("$RUNTIME_VENV/bin/python" python3.13 python3 python)

  local candidate
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  return 1
}

python_has_modules() {
  local python_bin="$1"
  "$python_bin" - <<'PY' >/dev/null 2>&1
mods = ['mlx', 'numpy', 'pyarrow', 'regex', 'requests', 'rustbpe', 'tiktoken']
for mod in mods:
    __import__(mod)
PY
}

write_env_file() {
  local mode="$1"
  local python_bin="$2"
  cat > "$ROOT_DIR/.autoresearch-runtime.env" <<EOF
AUTORESEARCH_RUNTIME_MODE=$mode
AUTORESEARCH_PYTHON=$python_bin
EOF
}

bootstrap_with_uv() {
  local python_bin="$1"
  local uv_log="$RUNTIME_LOG_DIR/uv-sync.log"

  if ! command -v uv >/dev/null 2>&1; then
    return 1
  fi

  if uv sync --python "$python_bin" >"$uv_log" 2>&1; then
    if [ -x "$ROOT_DIR/.venv/bin/python" ] && python_has_modules "$ROOT_DIR/.venv/bin/python"; then
      write_env_file "uv" "$ROOT_DIR/.venv/bin/python"
      echo "CHECK runtime_bootstrap_method=uv"
      echo "ARTIFACT runtime_bootstrap_log=$uv_log"
      return 0
    fi
  fi

  if grep -q "Attempted to create a NULL object" "$uv_log" 2>/dev/null; then
    echo "CHECK runtime_uv_panicked=true"
  fi
  echo "ARTIFACT runtime_bootstrap_log=$uv_log"
  return 1
}

bootstrap_with_venv() {
  local python_bin="$1"
  local pip_log="$RUNTIME_LOG_DIR/pip-install.log"

  if [ ! -x "$RUNTIME_VENV/bin/python" ]; then
    "$python_bin" -m venv "$RUNTIME_VENV"
  fi

  "$RUNTIME_VENV/bin/python" -m pip install -e . >"$pip_log" 2>&1

  if python_has_modules "$RUNTIME_VENV/bin/python"; then
    write_env_file "venv" "$RUNTIME_VENV/bin/python"
    echo "CHECK runtime_bootstrap_method=venv"
    echo "ARTIFACT runtime_bootstrap_log=$pip_log"
    return 0
  fi

  echo "ARTIFACT runtime_bootstrap_log=$pip_log"
  return 1
}

main() {
  local python_bin
  python_bin="$(find_python)" || {
    echo "CHECK runtime_python_found=false"
    exit 1
  }

  echo "CHECK runtime_python_found=true"
  echo "METRIC runtime_python_path=$python_bin"

  if python_has_modules "$python_bin"; then
    write_env_file "system" "$python_bin"
    echo "CHECK runtime_ready=true"
    echo "CHECK runtime_bootstrap_method=system"
    exit 0
  fi

  if [ "${AUTORESEARCH_DISABLE_UV:-0}" != "1" ] && bootstrap_with_uv "$python_bin"; then
    echo "CHECK runtime_ready=true"
    exit 0
  fi

  if bootstrap_with_venv "$python_bin"; then
    echo "CHECK runtime_ready=true"
    exit 0
  fi

  echo "CHECK runtime_ready=false"
  exit 1
}

main "$@"
