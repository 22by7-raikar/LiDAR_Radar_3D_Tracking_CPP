#!/usr/bin/env bash
# Regenerates the figures and the 3D animation in docs/images: runs the example
# simulations, then renders the plots from them and from output/bench
# (scripts/run_experiments.sh). Set PYTHON to choose the interpreter.
# The 3D still and GIF need the viewer build and an X display (xvfb-run works).
set -euo pipefail
cd "$(dirname "$0")/.."
py="${PYTHON:-python3}"
runs=output/runs
mkdir -p "$runs"
run() { ./build/fusion_run --config "$2" --seed 1 --out "$runs/$1" > /dev/null; }
run cut_in_imm config/experiments/3_cut_in/imm.yaml
run cut_in_ukf config/experiments/3_cut_in/ukf.yaml
run realistic config/realistic.yaml
"$py" scripts/render_figures.py --runs "$runs" --bench output/bench --out docs/images

viewer=build/fusion_viewer
[ -x "$viewer" ] || viewer=build-viewer/fusion_viewer
if [ -x "$viewer" ] && command -v xvfb-run > /dev/null; then
  xvfb() { xvfb-run -a -s "-screen 0 1280x720x24" "$viewer" "$@" > /dev/null 2>&1; }
  frames=output/viewer_frames
  mkdir -p "$frames"
  find "$frames" -name 'frame_*.png' -delete
  xvfb --config config/viewer_demo.yaml --record "$frames" --every 2
  "$py" scripts/frames_to_gif.py "$frames" docs/images/viewer_3d.gif --width 760 --fps 15
fi
