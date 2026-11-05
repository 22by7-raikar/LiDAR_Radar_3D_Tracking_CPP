#!/usr/bin/env python3
"""Renders an animation of one fusion_run output: the lidar scan in 3D, the
radar detections in range-bearing and range-Doppler views, and the fused
tracks against ground truth in a top view, all in the ego frame.

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
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

from render_figures import (BLUE, GREEN, INK, LIGHT, MUTED, ORANGE, PURPLE, RED,  # noqa: E402
                            Run, car_outline)

TRACK_COLORS = [BLUE, GREEN, PURPLE, "#c2457a", "#b8860b", "#4c8c2b"]
RADAR = ORANGE
LIDAR = "#5b6b82"
LANES = [-5.55, -1.85, 1.85, 5.55]
X_RANGE, Y_RANGE = (-28.0, 68.0), (-7.0, 7.0)
RADAR_RANGE = 75.0     # radar plots out to [m]
RADAR_PERSIST = 0.5    # radar detections fade out over [s]
RATE_LIMIT = 8.0       # colour scale of the range rate [m/s]


def load_points(path):
    """Scan times, and the returns (x, y, z in the ego frame) of each scan."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
    times = np.unique(data[:, 0])
    return times, {t: data[data[:, 0] == t, 1:4] for t in times}


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
    """One frame: the lidar scan, the radar detections in two radar views, and the
    fused tracks in a top view."""

    def __init__(self, run, scan_times, scans):
        self.run, self.scan_times, self.scans = run, scan_times, scans
        m = run.measurements
        self.radar = m[m["sensor"] == "radar"]
        self.est_times = np.unique(run.estimates["time"])
        tracks = sorted(set(run.estimates["track"].astype(int)))
        self.color = {tr: TRACK_COLORS[k % len(TRACK_COLORS)] for k, tr in enumerate(tracks)}

        self.fig = plt.figure(figsize=(12, 8.4), dpi=100)
        self.ax_lidar = self.fig.add_axes([0.02, 0.5, 0.48, 0.38])
        self.ax_polar = self.fig.add_axes([0.53, 0.53, 0.22, 0.33], projection="polar")
        self.ax_doppler = self.fig.add_axes([0.81, 0.53, 0.17, 0.33])
        self.ax = self.fig.add_axes([0.06, 0.1, 0.92, 0.3])
        self.fig.text(0.02, 0.965, "From raw sensor data to fused tracks", fontsize=13,
                      fontweight="bold", va="center")
        self.time_label = self.fig.text(0.98, 0.965, "", fontsize=11, family="monospace",
                                        ha="right", va="center")
        self.rate_map = plt.cm.ScalarMappable(norm=Normalize(-RATE_LIMIT, RATE_LIMIT),
                                              cmap="RdBu")
        cbar = self.fig.colorbar(self.rate_map, cax=self.fig.add_axes([0.6, 0.47, 0.3, 0.012]),
                                 orientation="horizontal")
        cbar.set_label("radar range rate [m/s]   red: closing, blue: opening", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        self.fig.legend(
            [Line2D([], [], color=LIDAR, marker=".", ls="", ms=6),
             Line2D([], [], color=RADAR, marker="o", ls="", ms=5),
             Patch(fc="none", ec=INK, ls="--"),
             Patch(fc=BLUE, alpha=0.45, ec=BLUE)],
            ["lidar returns", "radar detections (with clutter)", "true vehicle",
             "confirmed track: #id, estimated speed"],
            loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=8.5,
            frameon=False)

    def scan_at(self, t):
        k = np.searchsorted(self.scan_times, t + 1e-9) - 1
        return self.scans[self.scan_times[k]] if k >= 0 else np.zeros((0, 3))

    def radar_since(self, t, span):
        r = self.radar
        return r[(r["time"] > t - span) & (r["time"] <= t + 1e-9)]

    def frame(self, t):
        self.time_label.set_text(f"t = {t:4.1f} s")
        self.lidar_view(t)
        self.radar_views(t)
        self.top_view(t)

    def lidar_view(self, t):
        ax = self.ax_lidar
        ax.clear()
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.68, 0.22)
        ax.set_axis_off()
        cam = ChaseCamera()
        for lane in LANES:
            x = np.linspace(cam.position[0] + 3, X_RANGE[1], 60)
            u, v, _ = cam.project(x, np.full_like(x, lane), np.zeros_like(x))
            ax.plot(u, v, color=LIGHT, lw=1.0, ls="-" if abs(lane) > 5 else (0, (5, 5)))
        p = self.scan_at(t)
        u, v, depth = cam.project(p[:, 0], p[:, 1], p[:, 2])
        ax.scatter(u, v, c=np.hypot(p[:, 0], p[:, 1]), cmap="plasma_r", vmin=0, vmax=60,
                   s=1.2, lw=0, zorder=3)
        objects = self.run.objects
        for o in objects[np.abs(objects["time"] - self.scan_time(t)) < 1e-6]:
            draw_box(ax, cam, o["center_x"], o["center_y"], o["yaw"], GREEN)
        draw_box(ax, cam, 0.0, 0.0, 0.0, INK, fill=True)
        ax.set_title(f"LiDAR scan: {len(p):,} returns (colour: range); green: detected boxes",
                     fontsize=10, fontweight="bold", loc="left")

    def scan_time(self, t):
        k = np.searchsorted(self.scan_times, t + 1e-9) - 1
        return self.scan_times[k] if k >= 0 else -1.0

    def radar_views(self, t):
        r = self.radar_since(t, RADAR_PERSIST)
        age = t - r["time"]
        alpha = np.clip(1.0 - age / RADAR_PERSIST, 0.15, 1.0)
        colors = self.rate_map.to_rgba(r["z2"])
        colors[:, 3] = alpha

        ax = self.ax_polar
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.set_rlim(0, RADAR_RANGE)
        ax.set_rticks([20, 40, 60])
        ax.tick_params(labelsize=7)
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]),
                      ["ahead", "", "left", "", "behind", "", "right", ""])
        ax.scatter(r["z1"], r["z0"], s=18, color=colors, ec=MUTED, lw=0.4)
        ax.set_title("Radar: range and bearing", fontsize=10, fontweight="bold", pad=14)

        ax = self.ax_doppler
        ax.clear()
        ax.scatter(r["z2"], r["z0"], s=18, color=colors, ec=MUTED, lw=0.4)
        ax.axvline(0, color=LIGHT, lw=0.8)
        ax.set_xlim(-RATE_LIMIT * 1.5, RATE_LIMIT * 1.5)
        ax.set_ylim(0, RADAR_RANGE)
        ax.set_xlabel("range rate [m/s]", fontsize=8)
        ax.set_ylabel("range [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("Radar: range and Doppler", fontsize=10, fontweight="bold", loc="left")

    def top_view(self, t):
        run, ax = self.run, self.ax
        ax.clear()
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(*Y_RANGE)
        ax.set_aspect("equal")
        ax.set_xlabel("forward [m]", fontsize=9)
        ax.set_ylabel("left [m]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title("Fused output: tracks from both sensors (IMM filters, GNN association)",
                     fontsize=10, fontweight="bold", loc="left")
        ex, ey, eyaw, _, _ = run.pose(t)

        for lane in LANES:
            ly = (lane - ey) * np.cos(eyaw)
            style = "-" if abs(lane) > 5 else (0, (6, 6))
            ax.plot(X_RANGE, [ly, ly], color=LIGHT, lw=1.2, ls=style, zorder=0)

        p = self.scan_at(t)[::2]
        ax.scatter(p[:, 0], p[:, 1], s=2.5, color=LIDAR, lw=0, zorder=2)
        r = self.radar_since(t, 0.05)
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

    def render(self, times, path, fps):
        tmp = tempfile.mkdtemp()
        try:
            for k, t in enumerate(times):
                self.frame(t)
                self.fig.savefig(os.path.join(tmp, f"f{k:04d}.png"))
            write_gif(tmp, len(times), path, fps)
        finally:
            shutil.rmtree(tmp)


class ChaseCamera:
    """Pinhole camera behind and above the ego vehicle, pitched down, looking ahead."""

    def __init__(self, position=(-24.0, 0.0, 9.0), pitch_deg=13.0, focal=1.3):
        self.position = np.array(position)
        a = np.radians(pitch_deg)
        self.forward = np.array([np.cos(a), 0.0, -np.sin(a)])
        self.up = np.array([np.sin(a), 0.0, np.cos(a)])
        self.right = np.array([0.0, -1.0, 0.0])
        self.focal = focal

    def project(self, x, y, z):
        """Image coordinates (u right, v up) and depth; points behind the camera -> nan."""
        rel = np.stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)],
                       axis=-1) - self.position
        depth = rel @ self.forward
        depth = np.where(depth > 0.5, depth, np.nan)
        return (self.focal * (rel @ self.right) / depth, self.focal * (rel @ self.up) / depth,
                depth)


def draw_box(ax, cam, x, y, yaw, color, fill=False, height=1.5):
    """Vehicle box on the ground, as a wireframe or a solid, through the camera."""
    base = car_outline(x, y, yaw)
    bottom = cam.project(base[:, 0], base[:, 1], np.zeros(4))
    top = cam.project(base[:, 0], base[:, 1], np.full(4, height))
    if fill:
        for i in range(4):
            j = (i + 1) % 4
            quad = [(bottom[0][i], bottom[1][i]), (bottom[0][j], bottom[1][j]),
                    (top[0][j], top[1][j]), (top[0][i], top[1][i])]
            ax.add_patch(Polygon(quad, closed=True, fc=color, ec="white", lw=0.4, zorder=4))
        ax.add_patch(Polygon(np.c_[top[0], top[1]], closed=True, fc="#3a4048", ec="white",
                             lw=0.4, zorder=4))
        return
    for ring in (bottom, top):
        ax.plot(np.r_[ring[0], ring[0][0]], np.r_[ring[1], ring[1][0]], color=color, lw=1.1,
                zorder=4)
    for i in range(4):
        ax.plot([bottom[0][i], top[0][i]], [bottom[1][i], top[1][i]], color=color, lw=1.1,
                zorder=4)


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
        anim.frame(est_times[np.argmin(np.abs(est_times - args.still))])
        anim.fig.savefig(args.out, dpi=130)
        print("wrote", args.out)
        return
    step = max(1, int(round(1.0 / (args.fps * np.median(np.diff(est_times))))))
    times = est_times[(est_times >= args.start) & (est_times <= args.end)][::step]
    anim.render(times, args.out, args.fps)


if __name__ == "__main__":
    main()
