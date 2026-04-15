#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
current_commit=""

if current_commit="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null)"; then
  :
else
  current_commit="<short_commit>"
fi

printf 'repo_root=%s\n' "$repo_root"
printf 'results_path=%s\n' "$repo_root/results.tsv"
printf 'latest_log_path=%s\n' "$repo_root/run.log"
printf 'logs_dir=%s\n' "$repo_root/logs"
printf 'archive_example=%s\n' "$repo_root/logs/$current_commit.log"
