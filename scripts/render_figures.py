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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon  # noqa: E402

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

    group(0.2, 0.5, 2.9, 4.4, "SENSORS → detections + ego pose")
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
            "cost": rows["us_per_detection"],
        }
    return out


def velocity_rmse(row):
    """RMSE of the velocity vector, from the per-axis RMSEs."""
    return float(np.hypot(row["rmse"][2], row["rmse"][3]))


def fig_ablation(bench, out):
    """One bar per pipeline change: how much velocity error each change removes."""
    t = bench_table(os.path.join(bench, "1_ablation.csv"))
    steps = [("a_naive", "Naive tracker (textbook UKF)"),
             ("b_robust_ukf", "+ robust UKF numerics"),
             ("c_two_point_init", "+ two-detection start"),
             ("d_consistent_noise", "+ process noise tuned for NEES ≈ 2"),
             ("e_imm", "+ IMM (cruising / manoeuvring UKFs)")]
    steps = [(k, label) for k, label in steps if k in t]
    err = [velocity_rmse(t[k]) for k, _ in steps]
    nees = [np.mean(t[k]["nees"]) for k, _ in steps]
    y = np.arange(len(steps))[::-1]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    colors = [MUTED] + [BLUE] * (len(steps) - 2) + [GREEN]
    ax.barh(y, err, color=colors, height=0.62)
    for yi, e, n in zip(y, err, nees):
        ax.text(e + 0.006, yi, f"{e:.3f} m/s", va="center", fontsize=9.5, fontweight="bold")
        ax.text(0.505, yi, f"{n:.2f}", va="center", ha="right", fontsize=9.5,
                color=GREEN if abs(n - 2) < 0.1 else RED)
    ax.text(0.505, len(steps) - 0.35, "NEES (ideal 2)", ha="right", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, [label for _, label in steps], fontsize=9.5)
    ax.set_xlim(0, 0.51)
    ax.set_xlabel("velocity RMSE [m/s], highway scenario, mean of 100 runs")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"Each change and what it buys: velocity error "
                 f"{100 * (1 - err[-1] / err[0]):.0f} % lower, filter consistent")
    save(fig, out, "ablation.png")


def fig_filters(bench, out):
    """EKF, UKF and IMM on three scenarios, and what each costs."""
    groups = [("Highway (lidar + radar)", "2_filters.csv", {"ekf": "EKF", "ukf": "UKF",
                                                           "imm": "IMM"}),
              ("Cut-in and hard brake", "3_cut_in.csv", {"ekf": "EKF", "ukf": "UKF",
                                                         "imm": "IMM"}),
              ("Radar only, 10 Hz", "4_radar_rate.csv",
               {"radar_10hz_naive_ukf": "naive UKF", "radar_10hz_ekf": "EKF",
                "radar_10hz_ukf": "UKF", "radar_10hz_imm": "IMM"})]
    color = {"naive UKF": RED, "EKF": MUTED, "UKF": BLUE, "IMM": GREEN}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4),
                             gridspec_kw={"width_ratios": [3, 3, 4, 3], "wspace": 0.35})
    cost = {}
    for ax, (title, name, names) in zip(axes, groups):
        t = bench_table(os.path.join(bench, name))
        labels = [names[k] for k in names if k in t]
        values = [velocity_rmse(t[k]) for k in names if k in t]
        top = 1.25 * max(v for v in values if v < 5)
        for i, (label, v) in enumerate(zip(labels, values)):
            ax.bar(i, min(v, top), color=color[label], width=0.65)
            text = f"{v:.2f}" if v < top else f"{v:.0f} ↑\ndiverged"
            ax.text(i, min(v, top) * 1.02, text, ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=INK if v < top else RED)
        ax.set_xticks(range(len(labels)), labels, fontsize=9)
        ax.set_ylim(0, top * 1.15)
        ax.set_title(title, fontsize=10.5)
        for k in names:
            if k in t and names[k] not in cost and name == "2_filters.csv":
                cost[names[k]] = np.mean(t[k]["cost"])
    axes[0].set_ylabel("velocity RMSE [m/s]")
    t = bench_table(os.path.join(bench, "4_radar_rate.csv"))
    cost["naive UKF"] = np.mean(t["radar_10hz_naive_ukf"]["cost"]) if "radar_10hz_naive_ukf" in t \
        else np.nan
    order = [k for k in ("EKF", "UKF", "IMM") if k in cost]
    ax = axes[3]
    ax.bar(range(len(order)), [cost[k] for k in order], color=[color[k] for k in order],
           width=0.65)
    for i, k in enumerate(order):
        ax.text(i, cost[k] * 1.02, f"{cost[k]:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_xticks(range(len(order)), order, fontsize=9)
    ax.set_ylabel("µs per detection")
    ax.set_title("Cost (highway)", fontsize=10.5)
    fig.suptitle("Filters compared (100 runs each): EKF ≈ UKF, IMM is the most accurate, "
                 "the naive UKF diverges on radar alone", x=0.06, ha="left", y=1.04,
                 fontweight="bold", fontsize=12)
    save(fig, out, "filters.png")


def fig_imm(runs, out):
    """Mode probability and velocity error through the cut-in and the hard brake."""
    imm, ukf = runs["cut_in_imm"], runs["cut_in_ukf"]
    names = {0: "lead car", 1: "cutting-in car", 2: "slow car"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    events = []
    for v in imm.vehicles():
        s = imm.scored[imm.scored["vehicle"] == v]
        tracks = set(s["track"].astype(int))
        e = imm.estimates[np.isin(imm.estimates["track"].astype(int), list(tracks))]
        axes[0].plot(e["time"], e["mu1"], color=PALETTE[v], lw=1.6, label=names.get(v, str(v)))
        t = imm.truth[imm.truth["vehicle"] == v]
        dv = np.gradient(t["v"], t["time"])
        for mask, what in ((np.abs(t["yaw_rate"]) > 0.01, "lane change"),
                           (dv < -0.5, f"brakes {dv.min():.0f} m/s²")):
            if mask.any():
                events.append((t["time"][mask][0], t["time"][mask][-1],
                               f"{names.get(v, v)}\n{what}", PALETTE[v]))
    for k, (t0, t1, label, c) in enumerate(sorted(events)):
        for ax in axes:
            ax.axvspan(t0, t1, color=c, alpha=0.1, lw=0)
            ax.axvline(t0, color=c, lw=0.8, ls=":")
        axes[0].text(t0 + 0.1, 0.78 - 0.17 * k, label, fontsize=8.5, color=c, va="top",
                     fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    axes[0].set_ylabel("P(manoeuvring model)")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="upper left", ncol=3, fontsize=9)

    def speed_error(run):
        s = run.scored
        times = np.unique(s["time"])
        err = np.array([np.sqrt(np.mean(s["evx"][s["time"] == t] ** 2 +
                                        s["evy"][s["time"] == t] ** 2)) for t in times])
        k = 9
        return times, np.convolve(err, np.ones(k) / k, mode="same")

    for run, color, label in ((ukf, MUTED, "single UKF (noise sized for manoeuvres)"),
                              (imm, INK, "IMM")):
        t, e = speed_error(run)
        axes[1].plot(t, e, color=color, lw=1.5, label=label)
    axes[1].legend(loc="upper right", ncol=2, fontsize=9)
    axes[1].set_ylabel("velocity error [m/s]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylim(0, 0.8)
    axes[1].set_xlim(0, 20)
    fig.suptitle("The IMM switches to its manoeuvring model when a car manoeuvres, "
                 "and is more precise than one UKF the rest of the time", x=0.06, ha="left",
                 y=0.97,
                 fontweight="bold", fontsize=12)
    save(fig, out, "imm_cut_in.png")


def fig_scan(run, out):
    """How the lidar detector turns a scan into a vehicle position."""
    pts, obj, truth = run.points, run.objects, run.truth
    times = np.unique(obj["time"])
    frame = times[np.argmin(np.abs(times - 2.0))]
    p = pts[np.abs(pts["time"] - frame) < 1e-6]
    o = obj[np.abs(obj["time"] - frame) < 1e-6]
    tr = truth[np.abs(truth["time"] - frame) < 1e-6]
    rx, ry, ryaw = run.to_ego(tr["time"], tr["x"], tr["y"], tr["yaw"])
    # The car seen side-on (an L of returns) and the car seen only from behind.
    order = np.argsort(np.abs(ry) - 0.01 * np.abs(rx))[::-1]
    side, rear = order[0], int(np.argmax(rx))

    fig = plt.figure(figsize=(13, 3.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.25)
    for k, (i, title) in enumerate(((side, "Car beside the ego: an L of returns"),
                                    (rear, "Car ahead: only its rear is visible"))):
        ax = fig.add_subplot(grid[0, k])
        mine = p[p["vehicle"] == tr["vehicle"][i]]
        ax.scatter(mine["x"], mine["y"], s=3, color=BLUE, alpha=0.35, lw=0)
        ax.add_patch(Polygon(car_outline(rx[i], ry[i], ryaw[i]), closed=True, fc="none",
                             ec=INK, lw=1.2, ls="--"))
        for det in o[o["vehicle"] == tr["vehicle"][i]]:
            ax.plot(det["centroid_x"], det["centroid_y"], "x", color=RED, ms=11, mew=2.5)
            ax.add_patch(Polygon(car_outline(det["center_x"], det["center_y"], det["yaw"]),
                                 closed=True, fc=GREEN, alpha=0.18, ec=GREEN, lw=1.5))
            ax.plot(det["center_x"], det["center_y"], "o", color=GREEN, ms=8)
        ax.plot(rx[i], ry[i], "+", color=INK, ms=12, mew=1.5)
        ax.set_xlim(rx[i] - 3.8, rx[i] + 3.8)
        ax.set_ylim(ry[i] - 2.6, ry[i] + 2.6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for side_name in ("left", "bottom"):
            ax.spines[side_name].set_color(LIGHT)
        d = -np.array([rx[i], ry[i]]) / np.hypot(rx[i], ry[i])
        tip = (0.5 + 0.47 * d[0], 0.5 + 0.45 * d[1])
        tail = (0.5 + 0.33 * d[0], 0.5 + 0.33 * d[1])
        ax.annotate("", xy=tip, xytext=tail, xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
        ax.text(tail[0], tail[1] - 0.06, "to ego", transform=ax.transAxes, fontsize=8.5,
                color=MUTED, ha="center", va="top")
    fig.legend([Line2D([], [], color=BLUE, marker=".", ls="", ms=8),
                Line2D([], [], color=INK, ls="--"),
                Line2D([], [], color=RED, marker="x", ls="", mew=2.5, ms=9),
                Patch(fc=GREEN, alpha=0.3, ec=GREEN)],
               ["lidar returns (top view)", "true vehicle (+ centre)",
                "centroid of the returns", "fitted L-shape box (● centre)"],
               loc="upper left", ncol=2, fontsize=9, bbox_to_anchor=(0.12, 0.1))

    ax2 = fig.add_subplot(grid[0, 2])
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
    bins = np.linspace(0, 2.4, 49)
    ax2.hist(ec, bins=bins, color=RED, alpha=0.65, label=f"centroid: median {np.median(ec):.2f} m")
    ax2.hist(eb, bins=bins, color=GREEN, alpha=0.8,
             label=f"L-shape box: median {np.median(eb):.2f} m")
    ax2.set_xlabel("distance from the true centre [m]")
    ax2.set_ylabel("detections")
    ax2.set_title(f"Error over all {len(eb)} detections of a 20 s run", fontsize=10.5)
    ax2.legend(loc="upper right", fontsize=9)
    fig.suptitle("The centroid of the returns sits on the visible surface; the fitted box "
                 "recovers the real centre", x=0.1, ha="left", y=1.02, fontweight="bold",
                 fontsize=12)
    save(fig, out, "scan_detection.png")
    return float(np.median(eb)), float(np.median(ec))


def fig_clutter(bench, out):
    """Hit counting against the log-likelihood track score as radar clutter grows."""
    t = bench_table(os.path.join(bench, "7_clutter.csv"))
    rates = [0, 2, 5, 10, 20]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.25})
    for mgmt, color, label in (("hits", ORANGE, "hit count (M of N)"),
                               ("score", BLUE, "log-likelihood score")):
        rows = [t[f"clutter_{r:02d}_{mgmt}"] for r in rates]
        for ax, key in zip(axes, ("false", "ospa")):
            values = [r[key] for r in rows]
            ax.plot(rates, values, "o-", color=color, lw=2)
            ax.text(20.6, values[-1], label, color=color, fontsize=9, va="center",
                    fontweight="bold")
    axes[0].set_ylabel("false tracks at any time")
    axes[0].set_title("False tracks", fontsize=10.5)
    axes[1].set_ylabel("OSPA [m] (lower is better)")
    axes[1].set_title("Overall tracking error", fontsize=10.5)
    for ax in axes:
        ax.set_xlabel("radar clutter detections per frame")
        ax.set_xticks(rates)
        ax.set_xlim(-0.5, 27)
    fig.suptitle("Scoring tracks by likelihood keeps clutter from becoming tracks "
                 "(100 runs per point)", x=0.07, ha="left", y=1.03, fontweight="bold",
                 fontsize=12)
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
    fig_filters(args.bench, args.out)
    fig_imm(runs, args.out)
    box, centroid = fig_scan(runs["realistic"], args.out)
    print(f"scan detector median error: box {box:.3f} m, centroid {centroid:.3f} m")
    fig_clutter(args.bench, args.out)


if __name__ == "__main__":
    main()
