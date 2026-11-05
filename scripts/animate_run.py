#!/usr/bin/env python3
"""Renders a top-down animation of one fusion_run output: lidar returns,
radar detections, ground truth, confirmed tracks and the IMM mode
probabilities, all in the ego frame.

    python3 scripts/animate_run.py output/runs/demo docs/images/tracking.gif \
        --start 8 --end 18

Requires numpy, matplotlib and pillow; uses ffmpeg for a better palette when
it is installed.
"""

import argparse
import os
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from render_figures import (BLUE, GREEN, INK, LIGHT, MUTED, ORANGE, PURPLE, RED,  # noqa: E402
                            Run, car_outline)

TRACK_COLORS = [BLUE, GREEN, PURPLE, "#c2457a", "#b8860b", "#4c8c2b"]
RADAR = ORANGE
LIDAR = "#5b6b82"
LANES = [-5.55, -1.85, 1.85, 5.55]
X_RANGE, Y_RANGE = (-28.0, 68.0), (-7.0, 7.0)


def load_points(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2))
    times = np.unique(data[:, 0])
    return times, {t: data[data[:, 0] == t, 1:3] for t in times}


def manoeuvre(truth, vehicle, t):
    """What the vehicle is doing at t, from its true state, or ''."""
    rows = truth[truth["vehicle"] == vehicle]
    k = np.searchsorted(rows["time"], t)
    if k < 1 or k >= len(rows):
        return ""
    dv = (rows["v"][k] - rows["v"][k - 1]) / (rows["time"][k] - rows["time"][k - 1])
    if abs(rows["yaw_rate"][k]) > 0.01:
        return "changing lane"
    if dv < -0.5:
        return "braking"
    if dv > 0.5:
        return "accelerating"
    return ""


class Animator:
    def __init__(self, run, scan_times, scans):
        self.run, self.scan_times, self.scans = run, scan_times, scans
        m = run.measurements
        self.radar = m[m["sensor"] == "radar"]
        self.lidar_det = m[m["sensor"] == "lidar"]
        self.est_times = np.unique(run.estimates["time"])
        tracks = sorted(set(run.estimates["track"].astype(int)))
        self.color = {tr: TRACK_COLORS[k % len(TRACK_COLORS)] for k, tr in enumerate(tracks)}

        self.fig = plt.figure(figsize=(12, 5.0), dpi=100)
        self.ax = self.fig.add_axes([0.06, 0.42, 0.92, 0.44])
        self.ax_mu = self.fig.add_axes([0.06, 0.1, 0.92, 0.2])
        self.fig.text(0.06, 0.955, "LiDAR–radar tracking, top view from the ego vehicle",
                      fontsize=13, fontweight="bold", va="center")
        self.fig.legend(
            [Line2D([], [], color=LIDAR, marker=".", ls="", ms=6),
             Line2D([], [], color=RADAR, marker="o", ls="", ms=5),
             Patch(fc="none", ec=INK, ls="--"),
             Patch(fc=BLUE, alpha=0.45, ec=BLUE)],
            ["lidar returns", "radar detections (with clutter)", "true vehicle",
             "confirmed track: #id, estimated speed"],
            loc="center right", bbox_to_anchor=(0.98, 0.905), ncol=4, fontsize=8.5,
            frameon=False)

    def frame(self, t):
        run, ax = self.run, self.ax
        ax.clear()
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(*Y_RANGE)
        ax.set_aspect("equal")
        ax.set_xlabel("forward [m]", fontsize=9)
        ax.set_ylabel("left [m]", fontsize=9)
        ax.tick_params(labelsize=8)
        ex, ey, eyaw, _, _ = run.pose(t)

        for lane in LANES:
            ly = (lane - ey) * np.cos(eyaw)
            style = "-" if abs(lane) > 5 else (0, (6, 6))
            ax.plot(X_RANGE, [ly, ly], color=LIGHT, lw=1.2, ls=style, zorder=0)

        # Most recent lidar scan and the radar detections of the last 50 ms.
        k = np.searchsorted(self.scan_times, t + 1e-9) - 1
        if k >= 0:
            p = self.scans[self.scan_times[k]][::2]
            ax.scatter(p[:, 0], p[:, 1], s=2.5, color=LIDAR, lw=0, zorder=2)
        r = self.radar[(self.radar["time"] > t - 0.05) & (self.radar["time"] <= t + 1e-9)]
        if len(r):
            rx, ry = run.to_ego(np.full(len(r), t), r["wx"], r["wy"])
            ax.scatter(rx, ry, s=22, color=RADAR, ec="white", lw=0.5, zorder=5)

        truth = run.truth[np.abs(run.truth["time"] - t) < 1e-6]
        tx, ty, tyaw = run.to_ego(truth["time"], truth["x"], truth["y"], truth["yaw"])
        for i, row in enumerate(truth):
            ax.add_patch(Polygon(car_outline(tx[i], ty[i], tyaw[i]), closed=True, fc="none",
                                 ec=INK, lw=0.9, ls="--", zorder=3))
            what = manoeuvre(run.truth, row["vehicle"], t)
            if what and X_RANGE[0] < tx[i] < X_RANGE[1]:
                ax.text(tx[i], ty[i] - 1.9, what, ha="center", va="top", fontsize=8.5,
                        color=RED, fontweight="bold", zorder=6)

        est = run.estimates[np.abs(run.estimates["time"] - t) < 1e-6]
        if len(est):
            sx, sy, syaw = run.to_ego(est["time"], est["x"], est["y"], est["yaw"])
            for i, e in enumerate(est):
                c = self.color[int(e["track"])]
                ax.add_patch(Polygon(car_outline(sx[i], sy[i], syaw[i]), closed=True, fc=c,
                                     alpha=0.45, ec=c, lw=1.2, zorder=4))
                ax.text(sx[i], sy[i] + 1.5, f"#{int(e['track'])}  {e['v']:.1f} m/s",
                        ha="center", va="bottom", fontsize=8.5, color=c, fontweight="bold",
                        zorder=6, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))

        ax.add_patch(Polygon(car_outline(0, 0, 0), closed=True, fc=INK, ec=INK, zorder=4))
        ax.text(0, -1.6, "ego", ha="center", va="top", fontsize=8.5, color=INK, zorder=6)
        ax.set_title(f"t = {t:4.1f} s", fontsize=10, loc="left", family="monospace")
        self.mode_panel(t)

    def mode_panel(self, t):
        ax, est = self.ax_mu, self.run.estimates
        ax.clear()
        lo, hi = self.window
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("time [s]", fontsize=9)
        ax.set_title("IMM: probability of the manoeuvring model, per track", fontsize=10,
                     loc="left")
        last = []
        for tr, c in self.color.items():
            rows = est[(est["track"] == tr) & (est["time"] <= t + 1e-9) &
                       (est["time"] >= lo - 1)]
            if len(rows):
                ax.plot(rows["time"], rows["mu1"], color=c, lw=1.6)
                last.append((rows["mu1"][-1], rows["time"][-1], tr, c))
        y_prev = 2.0
        for mu, when, tr, c in sorted(last, reverse=True):
            y = min(mu, y_prev - 0.12)
            y_prev = y
            ax.text(when + 0.08, y, f"#{tr}", color=c, fontsize=8, va="center",
                    fontweight="bold")
        ax.axvline(t, color=MUTED, lw=0.8)

    def render(self, times, path, fps):
        self.window = (times[0], times[-1])
        tmp = tempfile.mkdtemp()
        try:
            for k, t in enumerate(times):
                self.frame(t)
                self.fig.savefig(os.path.join(tmp, f"f{k:04d}.png"))
            write_gif(tmp, len(times), path, fps)
        finally:
            shutil.rmtree(tmp)


def write_gif(frames_dir, count, path, fps):
    pattern = os.path.join(frames_dir, "f%04d.png")
    if shutil.which("ffmpeg"):
        palette = os.path.join(frames_dir, "palette.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", pattern, "-vf",
                        "palettegen=max_colors=64:stats_mode=diff", palette], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i",
                        pattern, "-i", palette, "-lavfi",
                        "paletteuse=dither=none:diff_mode=rectangle", path], check=True)
    else:
        from PIL import Image
        frames = [Image.open(pattern % k).convert("P", palette=Image.ADAPTIVE, colors=64)
                  for k in range(count)]
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=8.0)
    ap.add_argument("--end", type=float, default=18.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--still", type=float, default=None,
                    help="write one PNG at this time instead of the animation")
    args = ap.parse_args()

    run = Run(args.run)
    scan_times, scans = load_points(os.path.join(args.run, "scan_points.csv"))
    anim = Animator(run, scan_times, scans)
    est_times = anim.est_times
    if args.still is not None:
        anim.window = (args.start, args.end)
        anim.frame(est_times[np.argmin(np.abs(est_times - args.still))])
        anim.fig.savefig(args.out, dpi=130)
        print("wrote", args.out)
        return
    step = max(1, int(round(1.0 / (args.fps * np.median(np.diff(est_times))))))
    times = est_times[(est_times >= args.start) & (est_times <= args.end)][::step]
    anim.render(times, args.out, args.fps)


if __name__ == "__main__":
    main()
