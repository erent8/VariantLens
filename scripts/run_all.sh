#!/usr/bin/env bash
# Reproduce everything from scratch: data -> scores -> benchmark -> app bundle.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}; [ -x "$PY" ] || PY=.venv/bin/python
mkdir -p data/raw && cd data/raw
curl -sSO https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
curl -sSO https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/2023/variant_summary_2023-08.txt.gz
curl -sSO https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz
cd ../..
$PY scripts/01_build_benchmark.py
$PY scripts/02_fetch_sequences.py
PY=$PY bash scripts/run_scoring.sh
$PY scripts/04_train_variantlens.py
$PY scripts/04_benchmark.py
$PY scripts/05_build_app_data.py
$PY scripts/06_hero_figure.py TP53
