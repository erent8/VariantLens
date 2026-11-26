#!/usr/bin/env bash
# Full scoring run (~1.5 h on a laptop RTX 4060).
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
for m in esm2_650M esm2_8M esm2_35M esm2_150M; do
  $PY -u scripts/03_score_esm.py --model $m --mode wt
done
$PY -u scripts/03_score_esm.py --model esm2_650M --mode mm --sample 25000
