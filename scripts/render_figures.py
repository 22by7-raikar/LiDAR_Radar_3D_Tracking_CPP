#!/usr/bin/env python3
"""Renders the figures in docs/images from the CSV outputs of
fusion_run and fusion_bench. scripts/make_figures.sh produces those outputs
and calls this.

    python3 scripts/render_figures.py --runs output/runs --bench output/bench \
        --out docs/images

Requires numpy, matplotlib and pillow.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

INK = "#1f2328"
MUTED = "#8c959f"
LIGHT = "#d6dae0"
FAINT = "#eceef1"
RED = "#d1495b"
BLUE = "#2f6fdf"
GREEN = "#2a9d8f"
ORANGE = "#e07a1f"
PURPLE = "#7b5ea7"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, RED, "#b8860b", "#c2457a", "#4c8c2b"]
CAR_LENGTH, CAR_WIDTH = 4.6, 1.9

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


# ------------------------------------------------------------------ input ----
def read_csv(path):
    """CSV with a header row -> numpy record array (numeric columns as float)."""
    with open(path) as f:
        names = f.readline().strip().split(",")
        rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
    columns = []
    for values in (zip(*rows) if rows else [[] for _ in names]):
        try:
            columns.append(np.array(values, dtype=float))
        except ValueError:
            columns.append(np.array(values, dtype=str))
    return np.rec.fromarrays(columns, names=names)


class Run:
    """All CSV files written by one fusion_run --out DIR, with world → ego helpers."""

    def __init__(self, path):
        self.path = path
        for name in ("truth", "measurements", "estimates", "scored", "updates", "ego"):
            setattr(self, name, read_csv(os.path.join(path, name + ".csv")))
        points = os.path.join(path, "scan_points.csv")
        self.points = read_csv(points) if os.path.exists(points) else None
        objects = os.path.join(path, "scan_objects.csv")
        self.objects = read_csv(objects) if os.path.exists(objects) else None
        self._ego = {round(t, 6): (x, y, yaw, vx, vy) for t, x, y, yaw, vx, vy in
                     zip(self.ego["time"], self.ego["x"], self.ego["y"], self.ego["yaw"],
                         self.ego["vx"], self.ego["vy"])}

    def vehicles(self):
        return sorted(set(self.truth["vehicle"].astype(int)))

    def pose(self, t):
        return self._ego[round(t, 6)]

    def to_ego(self, t, x, y, yaw=None):
        """World position (and heading) at time t, in the ego frame at t."""
        ex, ey, eyaw, _, _ = (np.array([self.pose(tt)[k] for tt in np.atleast_1d(t)])
                              for k in range(5))
        c, s = np.cos(eyaw), np.sin(eyaw)
        dx, dy = np.asarray(x) - ex, np.asarray(y) - ey
        out = (c * dx + s * dy, -s * dx + c * dy)
        return out if yaw is None else out + (np.asarray(yaw) - eyaw,)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def car_outline(x, y, yaw, length=CAR_LENGTH, width=CAR_WIDTH):
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[length / 2, width / 2], [-length / 2, width / 2],
                        [-length / 2, -width / 2], [length / 2, -width / 2]])
    return corners @ np.array([[c, s], [-s, c]]) + [x, y]


# ---------------------------------------------------------------- figures ----
def fig_architecture(out):
    """Block diagram of the pipeline (portfolio and README)."""
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def block(x, y, w, h, title, lines, color, title_color="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color, ec="none", alpha=0.95))
        ax.text(x + 0.15, y + h - 0.22, title, color=title_color, fontsize=10.5,
                fontweight="bold", va="top")
        for k, line in enumerate(lines):
            ax.text(x + 0.15, y + h - 0.62 - 0.3 * k, line, color=title_color, fontsize=8.6,
                    va="top")

    def group(x, y, w, h, label):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="none", ec=LIGHT, lw=1.5, ls="--"))
        ax.text(x + 0.12, y + h + 0.08, label, color=MUTED, fontsize=9, fontweight="bold")

    def arrow(x0, y0, x1, y1, text=None, color=MUTED):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=1.6, shrinkA=2, shrinkB=2))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.1, text, ha="center", fontsize=8,
                    color=MUTED)

    group(0.2, 0.5, 2.9, 4.4, "SENSORS (ego frame)")
    block(0.35, 2.85, 2.6, 1.85, "LiDAR  10–30 Hz",
          ["64-beam ray-cast scan", "grid clusters + merge", "L-shape box on contour",
           "→ box centre (x, y)"], BLUE)
    block(0.35, 0.7, 2.6, 1.85, "Radar  20–30 Hz",
          ["range, bearing,", "relative range rate", "misses + clutter", "→ (ρ, φ, ρ̇)"], PURPLE)

    group(3.6, 0.5, 6.7, 4.4, "MULTI-TARGET TRACKER (world frame)")
    block(3.75, 3.35, 2.0, 1.35, "Predict", ["IMM mixing", "2 × CTRV UKF"], INK)
    block(6.0, 3.35, 2.0, 1.35, "Gate", ["χ² on NIS (99.9 %)", "cost d² + ln|S|"], INK)
    block(8.25, 3.35, 1.9, 1.35, "Assign", ["Hungarian (GNN)", "confirmed first"], INK)
    block(8.25, 0.7, 1.9, 2.1, "Update", ["unscented radar", "linear lidar", "(Joseph form)",
                                          "mode probs"], INK)
    block(6.0, 0.7, 2.0, 2.1, "Initialise", ["two detections", "≥ 0.2 s apart", "+ road prior",
                                             "full covariance"], INK)
    block(3.75, 0.7, 2.0, 2.1, "Track score", ["log-likelihood", "ratio: confirm,", "coast, delete",
                                               "(clutter-aware)"], INK)
    arrow(5.75, 4.02, 6.0, 4.02)
    arrow(8.0, 4.02, 8.25, 4.02)
    arrow(9.2, 3.35, 9.2, 2.8, None)
    arrow(8.25, 1.75, 8.0, 1.75)
    arrow(6.0, 1.75, 5.75, 1.75)
    arrow(4.75, 2.8, 4.75, 3.35)
    ax.text(9.25, 3.05, "matched", fontsize=8, color=MUTED)
    ax.text(6.05, 2.95, "unmatched detections", fontsize=8, color=MUTED)
    arrow(3.1, 3.75, 3.75, 3.95)
    ax.text(3.42, 4.55, "detections\n+ ego pose", ha="center", fontsize=8, color=MUTED)
    arrow(3.1, 1.6, 3.75, 3.65)

    group(10.8, 0.5, 2.0, 4.4, "OUTPUT")
    block(10.95, 2.85, 1.7, 1.85, "Tracks", ["id, position,", "velocity, heading,", "turn rate,",
                                             "covariance"], GREEN)
    block(10.95, 0.7, 1.7, 1.85, "Evaluation", ["RMSE, NEES, NIS", "OSPA, ID switches",
                                                "Monte Carlo"], ORANGE)
    arrow(10.3, 4.02, 10.95, 3.8)
    arrow(11.8, 2.85, 11.8, 2.55)
    ax.text(0.2, 5.35, "LiDAR–radar fusion tracking pipeline", fontsize=14, fontweight="bold")
    save(fig, out, "architecture.png")


def bench_table(path):
    data = read_csv(path)
    out = {}
    for name in dict.fromkeys(data["config"]):
        rows = data[data["config"] == name]
        out[name] = {
            "rmse": np.sqrt(np.mean(np.stack([rows[f"rmse_{k}"] ** 2 for k in
                                              ("px", "py", "vx", "vy")]), axis=1)),
            "p95": np.mean(rows["velocity_p95"]),
            "latency": np.mean(rows["latency"]),
            "nees": rows["nees_position"],
            "ospa": np.mean(rows["ospa"]),
            "false": np.mean(rows["false_tracks"]),
            "switches": np.mean(rows["id_switches"]),
        }
    return out


def fig_ablation(bench, out):
    t = bench_table(os.path.join(bench, "1_ablation.csv"))
    labels = {
        "a_naive": "naive UKF\ntracker",
        "b_robust_ukf": "+ robust\nUKF",
        "c_two_point_init": "+ two-point\ninit",
        "d_consistent_noise": "+ consistent\nprocess noise",
        "e_imm": "+ IMM",
    }
    names = [n for n in labels if n in t]
    x = np.arange(len(names))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 2]})
    w = 0.38
    vx = [t[n]["rmse"][2] for n in names]
    vy = [t[n]["rmse"][3] for n in names]
    ax.bar(x - w / 2, vx, w, color=BLUE, label="vx")
    ax.bar(x + w / 2, vy, w, color=ORANGE, label="vy")
    for i, n in enumerate(names):
        ax.text(i, max(vx[i], vy[i]) + 0.008, f"p95 {t[n]['p95']:.2f}", ha="center", fontsize=8,
                color=MUTED)
    ax.set_xticks(x, [labels[n] for n in names], fontsize=8)
    ax.set_ylabel("velocity RMSE after 1 s [m/s]")
    ax.set_title("Velocity error, one change at a time (100 seeds)")
    ax.legend(loc="upper right")

    parts = ax2.violinplot([t[n]["nees"] for n in names], positions=x, showmedians=True,
                           widths=0.7)
    for b in parts["bodies"]:
        b.set_facecolor(BLUE)
        b.set_alpha(0.35)
    for k in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[k].set_color(BLUE)
    ax2.axhline(2.0, color=GREEN, lw=1.2, ls="--")
    ax2.text(len(names) - 0.5, 2.05, "consistent = 2", color=GREEN, ha="right", fontsize=8,
             va="bottom")
    ax2.set_xticks(x, [labels[n] for n in names], fontsize=7, rotation=35, ha="right")
    ax2.set_ylabel("position NEES per run")
    ax2.set_title("Filter consistency")
    save(fig, out, "ablation.png")


def fig_imm(runs, out):
    imm, ukf = runs["cut_in_imm"], runs["cut_in_ukf"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True)
    names = {0: "lead (brakes 13 s)", 1: "cutter (cuts in 9.5 s)", 2: "slow"}
    for v in imm.vehicles():
        s = imm.scored[imm.scored["vehicle"] == v]
        tracks = set(s["track"].astype(int))
        e = imm.estimates[np.isin(imm.estimates["track"].astype(int), list(tracks))]
        axes[0].plot(e["time"], e["mu1"], color=PALETTE[v], lw=1.5, label=names.get(v, v))
        t = imm.truth[imm.truth["vehicle"] == v]
        dv = np.gradient(t["v"], t["time"])
        man = (np.abs(t["yaw_rate"]) > 0.02) | (np.abs(dv) > 0.2)
        for ax in axes:
            ax.fill_between(t["time"], 0, 1, where=man, color=PALETTE[v], alpha=0.1,
                            transform=ax.get_xaxis_transform(), lw=0)
    axes[0].set_ylabel("P(manoeuvring model)")
    axes[0].set_title("IMM mode probability (shaded: when each vehicle manoeuvres)")
    axes[0].legend(loc="upper left", ncol=3)
    axes[0].set_ylim(0, 1.02)

    def speed_error(run):
        s = run.scored
        times = np.unique(s["time"])
        err = np.array([np.sqrt(np.mean(s["evx"][s["time"] == t] ** 2 +
                                        s["evy"][s["time"] == t] ** 2)) for t in times])
        k = 9
        return times, np.convolve(err, np.ones(k) / k, mode="same")

    for run, color, label in ((ukf, ORANGE, "single UKF (tuned for manoeuvres)"),
                              (imm, BLUE, "IMM")):
        t, e = speed_error(run)
        axes[1].plot(t, e, color=color, lw=1.4, label=label)
    axes[1].set_ylabel("velocity error [m/s]\n(RMS over vehicles, smoothed)")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(loc="upper left")
    axes[1].set_ylim(bottom=0)
    save(fig, out, "imm_cut_in.png")


def fig_scan(run, out):
    pts, obj, truth = run.points, run.objects, run.truth
    times = np.unique(obj["time"])
    frame = times[np.argmin(np.abs(times - 2.0))]
    p = pts[np.abs(pts["time"] - frame) < 1e-6]
    o = obj[np.abs(obj["time"] - frame) < 1e-6]
    tr = truth[np.abs(truth["time"] - frame) < 1e-6]
    rx, ry, ryaw = run.to_ego(tr["time"], tr["x"], tr["y"], tr["yaw"])

    fig = plt.figure(figsize=(13.5, 4.2))
    grid = fig.add_gridspec(1, len(tr) + 1, width_ratios=[1] * len(tr) + [1.6])
    for k, row in enumerate(tr):
        ax = fig.add_subplot(grid[0, k])
        mine = p[p["vehicle"] == row["vehicle"]]
        ax.scatter(mine["x"], mine["y"], s=2, color=BLUE, alpha=0.35, lw=0)
        ax.add_patch(Polygon(car_outline(rx[k], ry[k], ryaw[k]), closed=True, fc="none", ec=INK,
                             lw=1, ls="--"))
        ax.plot(rx[k], ry[k], "+", color=INK, ms=10, mew=1.5)
        for det in o[o["vehicle"] == row["vehicle"]]:
            ax.plot(det["centroid_x"], det["centroid_y"], "x", color=RED, ms=9, mew=2)
            ax.plot(det["center_x"], det["center_y"], "o", color=GREEN, ms=7, mfc="none", mew=2)
        ax.set_xlim(rx[k] - 3.5, rx[k] + 3.5)
        ax.set_ylim(ry[k] - 3.5, ry[k] + 3.5)
        ax.set_aspect("equal")
        ax.set_title(f"car at ({rx[k]:.0f}, {ry[k]:.0f}) m", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        d = np.array([rx[k], ry[k]])
        d = -d / np.linalg.norm(d)
        ax.annotate("", xy=(rx[k] + 3.2 * d[0], ry[k] + 3.2 * d[1]),
                    xytext=(rx[k] + 2.4 * d[0], ry[k] + 2.4 * d[1]),
                    arrowprops=dict(arrowstyle="->", color=MUTED))
    fig.legend([Line2D([], [], color=BLUE, marker=".", ls=""),
                Line2D([], [], color=INK, ls="--"),
                Line2D([], [], color=INK, marker="+", ls="", ms=9),
                Line2D([], [], color=RED, marker="x", ls="", mew=2),
                Line2D([], [], color=GREEN, marker="o", ls="", mfc="none", mew=2)],
               ["lidar returns (top view)", "true outline", "true centre", "centroid of returns",
                "fitted box centre"], loc="lower left", ncol=5, fontsize=8,
               bbox_to_anchor=(0.02, -0.04))
    fig.text(0.02, 0.98, f"One ray-cast scan at t = {frame:.1f} s "
             "(grey arrow: towards the sensor)", fontweight="bold", fontsize=12, va="top")
    ax2 = fig.add_subplot(grid[0, -1])

    ox, oy = run.to_ego(truth["time"], truth["x"], truth["y"])
    tmap = {(round(t, 6), int(v)): (x, y) for t, v, x, y in zip(truth["time"], truth["vehicle"],
                                                                 ox, oy)}
    ec, eb = [], []
    for row in obj:
        key = (round(row["time"], 6), int(row["vehicle"]))
        if key in tmap:
            x, y = tmap[key]
            ec.append(np.hypot(row["centroid_x"] - x, row["centroid_y"] - y))
            eb.append(np.hypot(row["center_x"] - x, row["center_y"] - y))
    bins = np.linspace(0, 3.0, 61)
    ax2.hist(ec, bins=bins, color=RED, alpha=0.6, label=f"centroid (median {np.median(ec):.2f} m)")
    ax2.hist(eb, bins=bins, color=GREEN, alpha=0.7,
             label=f"box centre (median {np.median(eb):.2f} m)")
    ax2.set_xlabel("position error of the detection [m]")
    ax2.set_ylabel("detections")
    ax2.set_title(f"All {len(eb)} detections of the run")
    ax2.legend()
    save(fig, out, "scan_detection.png")
    return float(np.median(eb)), float(np.median(ec))


def fig_clutter(bench, out):
    t = bench_table(os.path.join(bench, "7_clutter.csv"))
    rates = [0, 2, 5, 10, 20]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for mgmt, color, label in (("score", BLUE, "log-likelihood score"),
                               ("hits", ORANGE, "hit count (M = 4)")):
        rows = [t[f"clutter_{r:02d}_{mgmt}"] for r in rates]
        axes[0].plot(rates, [r["ospa"] for r in rows], "o-", color=color, label=label)
        axes[1].plot(rates, [r["false"] for r in rows], "o-", color=color, label=label)
        axes[2].plot(rates, [r["switches"] for r in rows], "o-", color=color, label=label)
    axes[0].set_ylabel("mean OSPA [m]")
    axes[1].set_ylabel("false tracks per scored time")
    axes[2].set_ylabel("ID switches per run")
    for ax in axes:
        ax.set_xlabel("radar clutter per frame")
        ax.set_xticks(rates)
    axes[0].legend()
    axes[0].set_title("Track management under clutter")
    save(fig, out, "clutter.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="output/runs")
    ap.add_argument("--bench", default="output/bench")
    ap.add_argument("--out", default="docs/images")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {name: Run(os.path.join(args.runs, name)) for name in
            ("cut_in_imm", "cut_in_ukf", "realistic")}
    fig_architecture(args.out)
    fig_ablation(args.bench, args.out)
    fig_imm(runs, args.out)
    box, centroid = fig_scan(runs["realistic"], args.out)
    print(f"scan detector median error: box {box:.3f} m, centroid {centroid:.3f} m")
    fig_clutter(args.bench, args.out)


if __name__ == "__main__":
    main()
