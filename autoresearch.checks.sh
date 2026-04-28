#!/bin/bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_CMD=./scripts/ensure-runtime.sh
LINT_CMD=''
CHECK_CMD=''
TEST_CMD=''

run_check() {
  local label="$1"
  local cmd="$2"
  local condensed="${cmd//[[:space:]]/}"

  if [ -z "$condensed" ] || [ "$cmd" = "not configured" ]; then
    echo "CHECK ${label}_configured=false"
    return 0
  fi

  echo "CHECK ${label}_configured=true"
  if bash -lc "$cmd"; then
    echo "CHECK ${label}_passed=true"
  else
    echo "CHECK ${label}_passed=false"
    return 1
  fi
}

run_check build "$BUILD_CMD"
run_check lint "$LINT_CMD"
run_check check "$CHECK_CMD"
run_check test "$TEST_CMD"
