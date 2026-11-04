#!/usr/bin/env bash
# Runs every experiment group in config/experiments with fusion_bench and
# writes per-run CSVs and summary tables to output/bench.
#
#   scripts/run_experiments.sh [RUNS]      (default 100 seeds per config)
set -euo pipefail
cd "$(dirname "$0")/.."
runs="${1:-100}"
mkdir -p output/bench
for group in config/experiments/*/; do
  name="$(basename "$group")"
  echo "== $name"
  ./build/fusion_bench --runs "$runs" --out "output/bench/$name.csv" "$group"*.yaml \
    | tee "output/bench/$name.txt"
done
