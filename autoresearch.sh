#!/bin/bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "CHECK autoresearch_standard_version=1"
echo "CHECK autoresearch_runner_ready=true"
echo "ARTIFACT autoresearch_doc=$SCRIPT_DIR/autoresearch.md"

if [ "${AUTORESEARCH_SKIP_BENCH:-0}" = "1" ]; then
  echo "CHECK autoresearch_bench_skipped=true"
elif [ -x "./autoresearch.bench.sh" ]; then
  ./autoresearch.bench.sh
else
  echo "CHECK autoresearch_bench_present=false"
fi

if [ "${AUTORESEARCH_SKIP_CHECKS:-0}" = "1" ]; then
  echo "CHECK autoresearch_checks_skipped=true"
elif [ -x "./autoresearch.checks.sh" ]; then
  ./autoresearch.checks.sh
else
  echo "CHECK autoresearch_checks_present=false"
fi
