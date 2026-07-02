#!/bin/bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin${PATH:+:$PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CACHE_FILE="$HOME/.cache/autoresearch/tokenizer/token_bytes.npy"
RESULTS_FILE="$SCRIPT_DIR/results.tsv"
RUN_LOG="$SCRIPT_DIR/run.log"

if [ -f "$CACHE_FILE" ]; then
  echo "CHECK tokenizer_cache_present=true"
else
  echo "CHECK tokenizer_cache_present=false"
fi

if [ -f "$RUN_LOG" ]; then
  echo "CHECK run_log_present=true"
  echo "ARTIFACT run_log=$RUN_LOG"
else
  echo "CHECK run_log_present=false"
fi

if [ -f "$RESULTS_FILE" ]; then
  rows="$(awk 'END{print NR}' "$RESULTS_FILE")"
  if [ "$rows" -gt 0 ]; then
    rows=$((rows - 1))
  fi
  echo "CHECK results_file_present=true"
  echo "METRIC results_rows=$rows"
  echo "ARTIFACT results_tsv=$RESULTS_FILE"
else
  echo "CHECK results_file_present=false"
  echo "METRIC results_rows=0"
fi
