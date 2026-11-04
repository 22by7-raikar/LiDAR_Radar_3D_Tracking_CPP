#!/usr/bin/env python3
"""Prints one Markdown table per experiment group from the CSV files that
fusion_bench writes (scripts/run_experiments.sh puts them in output/bench).

    python3 scripts/results_tables.py output/bench
"""

import csv
import math
import os
import sys
from collections import OrderedDict


def summarise(rows):
    n = len(rows)
    mean = lambda k: sum(float(r[k]) for r in rows) / n  # noqa: E731
    rmse = lambda k: math.sqrt(sum(float(r[k]) ** 2 for r in rows) / n)  # noqa: E731
    return OrderedDict([
        ("px", f"{rmse('rmse_px'):.3f}"), ("py", f"{rmse('rmse_py'):.3f}"),
        ("vx", f"{rmse('rmse_vx'):.3f}"), ("vy", f"{rmse('rmse_vy'):.3f}"),
        ("v p95", f"{mean('velocity_p95'):.2f}"),
        ("latency [s]", f"{mean('latency'):.2f}"),
        ("NEES", f"{mean('nees_position'):.2f}"),
        ("NIS lidar", f"{mean('nis_lidar'):.2f}" if mean('nis_lidar') else "–"),
        ("NIS radar", f"{mean('nis_radar'):.2f}" if mean('nis_radar') else "–"),
        ("OSPA [m]", f"{mean('ospa'):.3f}"),
        ("ID sw.", f"{mean('id_switches'):.2f}"),
        ("false trk.", f"{mean('false_tracks'):.3f}"),
        ("runs", str(n)),
    ])


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "output/bench"
    for name in sorted(f for f in os.listdir(folder) if f.endswith(".csv")):
        with open(os.path.join(folder, name)) as f:
            rows = list(csv.DictReader(f))
        groups = OrderedDict()
        for r in rows:
            groups.setdefault(r["config"], []).append(r)
        print(f"### {name[:-4]}\n")
        header = None
        for config, rs in groups.items():
            s = summarise(rs)
            if header is None:
                header = ["config"] + list(s)
                print("| " + " | ".join(header) + " |")
                print("|" + "---|" * len(header))
            print(f"| `{config}` | " + " | ".join(s.values()) + " |")
        print()


if __name__ == "__main__":
    main()
