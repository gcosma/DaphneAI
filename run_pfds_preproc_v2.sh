#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="output/pfds_v2_preproc"
mkdir -p "$OUT_DIR"

while IFS= read -r -d '' pdf; do
  echo "==> $pdf"
  python -m tools.preproc_v2_preview --pdf "$pdf" --output-dir "$OUT_DIR"
done < <(find pfds -type f -name '*.pdf' -print0)

