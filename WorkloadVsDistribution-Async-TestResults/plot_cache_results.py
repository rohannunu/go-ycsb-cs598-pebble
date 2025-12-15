#!/usr/bin/env python3
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt

# -----------------------------------------
# Config
# -----------------------------------------
PATTERN = "WorkloadVsDistribution_wl*_*.out"
CACHE_ORDER = ["none", "lru", "lfu", "density", "detox"]
WORKLOADS = ["A", "B", "C"]
DIST_FILTER = "zipfian"   # only use zipfian runs
THREAD_FILTER = "t32"     # only use 32-thread runs

# -----------------------------------------
# Parse one YCSB .out file
# -----------------------------------------
TOTAL_RE = re.compile(
    r"TOTAL\s+-.*OPS:\s*([\d.]+).*Avg\(us\):\s*([\d.]+).*95th\(us\):\s*([\d.]+).*99th\(us\):\s*([\d.]+)"
)

FNAME_RE = re.compile(
    r"WorkloadVsDistribution_wl([ABC])_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_t(\d+)_([A-Z]+)\.out"
)


def parse_file(path):
    """
    Returns (ops, avg_us, p95_us, p99_us) from the FINAL TOTAL line.
    """
    last_total = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("TOTAL  -"):
                last_total = line.strip()

    if last_total is None:
        raise ValueError(f"No TOTAL line found in {path}")

    m = TOTAL_RE.search(last_total)
    if not m:
        raise ValueError(f"Could not parse TOTAL line in {path}:\n{last_total}")

    ops, avg, p95, p99 = map(float, m.groups())
    return ops, avg, p95, p99


def annotate_bars(ax, rects, values, fmt="{:.0f}"):
    """
    Put text labels on top of each bar.
    """
    for rect, val in zip(rects, values):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )


# -----------------------------------------
# Collect data
# -----------------------------------------
# data[(wl, cache)] = {"ops": ..., "avg": ..., "p95": ..., "p99": ...}
data = {}

for path in glob.glob(PATTERN):
    fname = os.path.basename(path)
    m = FNAME_RE.match(fname)
    if not m:
        continue

    wl, cache, dist, threads, mode = m.groups()

    # Filter for zipfian & specific thread count if desired
    if dist != DIST_FILTER:
        continue
    if f"t{threads}" != THREAD_FILTER:
        continue

    wl = wl.upper()
    cache = cache.lower()

    try:
        ops, avg, p95, p99 = parse_file(path)
    except Exception as e:
        print(f"[WARN] Skipping {fname}: {e}")
        continue

    data[(wl, cache)] = {"ops": ops, "avg": avg, "p95": p95, "p99": p99}

# -----------------------------------------
# Sanity check
# -----------------------------------------
if not data:
    print("No matching data found. Check PATTERN / filters at top of script.")
    exit(1)

# -----------------------------------------
# Plot per-workload comparisons
# -----------------------------------------
os.makedirs("plots", exist_ok=True)

for wl in WORKLOADS:
    # Only include caches we actually have data for
    caches = [c for c in CACHE_ORDER if (wl, c) in data]
    if not caches:
        print(f"[INFO] No data for workload {wl}, skipping.")
        continue

    x = list(range(len(caches)))
    ops_vals = [data[(wl, c)]["ops"] for c in caches]
    avg_vals = [data[(wl, c)]["avg"] for c in caches]
    p99_vals = [data[(wl, c)]["p99"] for c in caches]

    # ----- Text summary -----
    print(f"\n=== Workload {wl} (zipfian, 32 threads) ===")
    print(f"{'cache':10s} {'OPS':>10s} {'avg(us)':>10s} {'p99(us)':>10s}")
    for c in caches:
        d = data[(wl, c)]
        print(
            f"{c:10s} "
            f"{d['ops']:10.1f} "
            f"{d['avg']:10.0f} "
            f"{d['p99']:10.0f}"
        )

    # ----- Plots -----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Throughput
    bars0 = axes[0].bar(x, ops_vals)
    axes[0].set_title(f"Workload {wl} – Throughput (OPS)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(caches, rotation=45, ha="right")
    axes[0].set_ylabel("OPS")
    annotate_bars(axes[0], bars0, ops_vals, fmt="{:.0f}")

    # Average latency
    bars1 = axes[1].bar(x, avg_vals)
    axes[1].set_title(f"Workload {wl} – Avg latency (µs)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(caches, rotation=45, ha="right")
    axes[1].set_ylabel("µs")
    annotate_bars(axes[1], bars1, avg_vals, fmt="{:.0f}")

    # 99th percentile latency
    bars2 = axes[2].bar(x, p99_vals)
    axes[2].set_title(f"Workload {wl} – 99th % latency (µs)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(caches, rotation=45, ha="right")
    axes[2].set_ylabel("µs")
    annotate_bars(axes[2], bars2, p99_vals, fmt="{:.0f}")

    fig.suptitle(f"Workload {wl} (zipfian, 32 threads): Cache Comparison", y=1.05)
    fig.tight_layout()

    out_png = os.path.join("plots", f"wl{wl}_zipfian_t32_cache_comparison.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[INFO] Saved plot for workload {wl} -> {out_png}")

print("\nDone. Check the 'plots/' directory for PNGs.")
