#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  exit 0
fi

paths=()
for path in "$@"; do
  # pre-commit passes paths from the repo root; Ty runs in the spyx project.
  if [[ "$path" == spyx/* ]]; then
    paths+=("${path#spyx/}")
  else
    paths+=("$path")
  fi
done

uv --directory spyx run --with ty ty check "${paths[@]}"
