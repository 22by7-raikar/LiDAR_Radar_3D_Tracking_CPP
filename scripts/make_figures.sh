#!/usr/bin/env bash
# Regenerates docs/images: runs the example simulations, renders the plots
# from them and from output/bench (scripts/run_experiments.sh), and renders the
# top-down tracking animation. Set PYTHON to choose the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
py="${PYTHON:-python3}"
runs=output/runs
mkdir -p "$runs"
run() { ./build/fusion_run --config "$2" --seed 1 --out "$runs/$1" > /dev/null; }
run cut_in_imm config/experiments/3_cut_in/imm.yaml
run cut_in_ukf config/experiments/3_cut_in/ukf.yaml
run realistic config/realistic.yaml
run demo config/demo.yaml
"$py" scripts/render_figures.py --runs "$runs" --bench output/bench --out docs/images
"$py" scripts/animate_run.py "$runs/demo" docs/images/tracking.gif --start 8 --end 18
