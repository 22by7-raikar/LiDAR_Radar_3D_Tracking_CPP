#!/usr/bin/env python3
"""Renders what each sensor contributes: the same traffic tracked with LiDAR
only, radar only, and both, one row each, drawn from three fusion_run outputs
of the same seed. Each row shows that sensor's detections with their
measurement uncertainty and the resulting tracks with their 2σ position
ellipses; the bottom panel compares the velocity error of the three.

    python3 scripts/animate_sensors.py output/runs/demo_lidar_only \
        output/runs/demo_radar_only output/runs/demo docs/images/sensors.gif

Requires numpy, matplotlib and pillow (ffmpeg optional).
"""

import argparse
import os
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Patch, Polygon  # noqa: E402

from animate_run import LANES, LIDAR, RADAR, load_points, write_gif  # noqa: E402
from render_figures import BLUE, GREEN, INK, LIGHT, MUTED, PURPLE, Run, car_outline  # noqa: E402

X_RANGE, Y_RANGE = (-28.0, 68.0), (-7.0, 7.0)
# Measurement noise of config/default.yaml (noise.lidar_std, noise.radar_*_std).
LIDAR_STD, RANGE_STD, BEARING_STD = 0.15, 0.3, 0.03
ROWS = [("LiDAR only", "#4c5d78",
         "precise position, but speed only from how the position changes"),
        ("Radar only", "#c8641a",
         "measures closing speed directly; coarse bearing blurs it sideways"),
        ("LiDAR + radar fused", GREEN, "position from the lidar, speed from the radar")]
VEHICLE_COLORS = [BLUE, PURPLE, "#c2457a", "#b8860b", "#4c8c2b"]
RATE_SCALE, RATE_MAX = 0.5, 5.0  # range-rate arrow: metres per m/s, longest arrow [m]


def errors(run, a, b):
    """RMS over the scored vehicles of the error vector (a, b) at each time, smoothed."""
    s = run.scored
    times = np.unique(s["time"])
    err = np.array([np.sqrt(np.mean(s[a][s["time"] == t] ** 2 + s[b][s["time"] == t] ** 2))
                    for t in times])
    k = 7
    return times, np.convolve(np.pad(err, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def run_rmse(run, a, b):
    s = run.scored[run.scored["time"] >= 1.0]
    return float(np.sqrt(np.mean(s[a] ** 2 + s[b] ** 2)))


def track_vehicles(run):
    """Track identity -> the vehicle it was scored against most often."""
    out = {}
    for tr in set(run.scored["track"].astype(int)):
        v = run.scored["vehicle"][run.scored["track"] == tr].astype(int)
        out[tr] = int(np.bincount(v).argmax())
    return out


def ellipse(ax, x, y, cov, color, **kw):
    """2σ ellipse of a 2×2 covariance."""
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-9)
    angle = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse((x, y), 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]), angle=angle, fc="none",
                         ec=color, **kw))


class SensorAnimator:
    def __init__(self, runs, scans):
        self.runs, self.scans = runs, scans
        self.pos_err = [errors(r, "ex", "ey") for r in runs]
        self.vel_err = [errors(r, "evx", "evy") for r in runs]
        self.summary = [(run_rmse(r, "ex", "ey"), run_rmse(r, "evx", "evy")) for r in runs]
        self.owner = [track_vehicles(r) for r in runs]
        self.fig = plt.figure(figsize=(12, 9.0), dpi=100)
        self.axes = [self.fig.add_axes([0.06, 0.745 - 0.2 * k, 0.92, 0.15]) for k in range(3)]
        self.ax_pos = self.fig.add_axes([0.06, 0.08, 0.42, 0.16])
        self.ax_vel = self.fig.add_axes([0.56, 0.08, 0.42, 0.16])
        self.fig.text(0.06, 0.975, "What each sensor sees, and what fusing them gives",
                      fontsize=13, fontweight="bold", va="center")
        self.fig.legend(
            [Line2D([], [], color=LIDAR, marker=".", ls="", ms=6),
             Line2D([], [], color=RADAR, marker="o", ls="", ms=5),
             Line2D([], [], color=RADAR, lw=1.8),
             Patch(fc="none", ec=INK, ls="--"),
             Patch(fc=BLUE, alpha=0.35, ec=BLUE),
             Patch(fc="none", ec=INK, lw=1.5)],
            ["lidar returns", "radar detection (2σ)", "radar range rate (closing speed)", "true vehicle",
             "track", "track 2σ position"],
            loc="center left", bbox_to_anchor=(0.05, 0.945), ncol=6, fontsize=8.5,
            frameon=False)
        self.time_label = self.fig.text(0.98, 0.975, "", fontsize=11, family="monospace",
                                        ha="right", va="center")

    def frame(self, t):
        for k, (ax, run, (name, color, note)) in enumerate(zip(self.axes, self.runs, ROWS)):
            self.row(ax, run, self.scans[k], t, k)
            pos, vel = self.summary[k]
            ax.set_title(f"{name}: {note}", fontsize=10, loc="left", color=color,
                         fontweight="bold")
            ax.set_title(f"20 s RMSE: position {pos:.2f} m, velocity {vel:.2f} m/s",
                         fontsize=9, loc="right", color=color)
        self.time_label.set_text(f"t = {t:4.1f} s")
        self.error_panel(self.ax_pos, self.pos_err, t, "Position error [m]", 1.0)
        self.error_panel(self.ax_vel, self.vel_err, t, "Velocity error [m/s]", 1.5)

    def row(self, ax, run, scans, t, k):
        ax.clear()
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(*Y_RANGE)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7.5)
        ax.set_ylabel("left [m]", fontsize=8)
        if k == 2:
            ax.set_xlabel("forward [m]", fontsize=8)
        ex, ey, eyaw, _, _ = run.pose(t)
        for lane in LANES:
            ly = (lane - ey) * np.cos(eyaw)
            ax.plot(X_RANGE, [ly, ly], color=LIGHT, lw=1.0,
                    ls="-" if abs(lane) > 5 else (0, (6, 6)), zorder=0)

        if scans is not None:
            scan_times, points = scans
            i = np.searchsorted(scan_times, t + 1e-9) - 1
            if i >= 0:
                p = points[scan_times[i]][::2]
                ax.scatter(p[:, 0], p[:, 1], s=2, color=LIDAR, lw=0, zorder=5)

        m = run.measurements
        r = m[(m["sensor"] == "radar") & (m["time"] > t - 0.05) & (m["time"] <= t + 1e-9)]
        for rho, phi, rate in zip(r["z0"], r["z1"], r["z2"]):
            x, y = rho * np.cos(phi), rho * np.sin(phi)
            ax.add_patch(Ellipse((x, y), 4 * RANGE_STD, 4 * rho * BEARING_STD,
                                 angle=np.degrees(phi), fc=RADAR, alpha=0.18, ec=RADAR, lw=0.6,
                                 zorder=3))
            ax.plot(x, y, "o", color=RADAR, ms=3.5, zorder=5)
            length = np.clip(rate * RATE_SCALE, -RATE_MAX, RATE_MAX)
            ax.plot([x, x + length * np.cos(phi)], [y, y + length * np.sin(phi)], color=RADAR,
                    lw=1.8, zorder=5)

        truth = run.truth[np.abs(run.truth["time"] - t) < 1e-6]
        tx, ty, tyaw = run.to_ego(truth["time"], truth["x"], truth["y"], truth["yaw"])
        for i in range(len(truth)):
            ax.add_patch(Polygon(car_outline(tx[i], ty[i], tyaw[i]), closed=True, fc="none",
                                 ec=INK, lw=0.8, ls="--", zorder=3))

        est = run.estimates[np.abs(run.estimates["time"] - t) < 1e-6]
        if len(est):
            sx, sy, syaw = run.to_ego(est["time"], est["x"], est["y"], est["yaw"])
            c, s = np.cos(eyaw), np.sin(eyaw)
            rot = np.array([[c, s], [-s, c]])
            for i, e in enumerate(est):
                v = self.owner[k].get(int(e["track"]), -1)
                color = VEHICLE_COLORS[v % len(VEHICLE_COLORS)] if v >= 0 else MUTED
                ax.add_patch(Polygon(car_outline(sx[i], sy[i], syaw[i]), closed=True, fc=color,
                                     alpha=0.25, ec="none", zorder=4))
                cov = rot @ np.array([[e["pxx"], e["pxy"]], [e["pxy"], e["pyy"]]]) @ rot.T
                ellipse(ax, sx[i], sy[i], cov, INK, lw=1.4, zorder=7)
        ax.add_patch(Polygon(car_outline(0, 0, 0), closed=True, fc=INK, ec=INK, zorder=4))

    def error_panel(self, ax, series, t, title, top):
        ax.clear()
        ax.set_xlim(*self.window)
        ax.set_ylim(0, top)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("time [s]", fontsize=8)
        ax.set_title(title + ", RMS over vehicles", fontsize=10, loc="left")
        for (times, err), (name, color, _) in zip(series, ROWS):
            keep = (times >= self.window[0] - 1) & (times <= t + 1e-9)
            ax.plot(times[keep], err[keep], color=color, lw=1.6, label=name)
        ax.axvline(t, color=MUTED, lw=0.8)
        ax.legend(loc="upper left", ncol=3, fontsize=8)

    def render(self, times, path, fps):
        tmp = tempfile.mkdtemp()
        try:
            for k, t in enumerate(times):
                self.frame(t)
                self.fig.savefig(os.path.join(tmp, f"f{k:04d}.png"))
            write_gif(tmp, len(times), path, fps)
        finally:
            shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lidar_run")
    ap.add_argument("radar_run")
    ap.add_argument("fused_run")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=8.0)
    ap.add_argument("--end", type=float, default=18.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--still", type=float, default=None,
                    help="write one PNG at this time instead of the animation")
    args = ap.parse_args()

    paths = [args.lidar_run, args.radar_run, args.fused_run]
    runs = [Run(p) for p in paths]
    scans = []
    for p in paths:
        f = os.path.join(p, "scan_points.csv")
        scans.append(load_points(f) if os.path.exists(f) else None)
    anim = SensorAnimator(runs, scans)
    anim.window = (args.start, args.end)
    est_times = np.unique(runs[2].estimates["time"])
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
