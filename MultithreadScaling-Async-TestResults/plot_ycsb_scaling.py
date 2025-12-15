#!/usr/bin/env python3
import re
import sys
import os

import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt

# =======================
# Config: shared y-limits
# =======================
THROUGHPUT_YLIM = (0, 100000)   # ops/sec, adjust if needed
P99_LATENCY_YLIM = (0, 20000)   # µs, adjust if needed

# Regex to grab threadcount from the properties block
THREAD_RE = re.compile(r'"threadcount"="(\d+)"')

# Regex to grab the base metrics lines (READ / TOTAL / UPDATE)
LINE_RE = re.compile(
    r'^\s*(READ|TOTAL|UPDATE)\s+- Takes\(s\):\s*([\d.]+), '
    r'Count:\s*([\d]+), OPS:\s*([\d.]+), Avg\(us\):\s*([\d]+)'
)

# Regex to grab p99 latency from the same line
P99_RE = re.compile(r'99th\(us\):\s*([\d.]+)')


def parse_log(path):
    """
    Parse a single YCSB-style log file.

    Returns:
        threadcount (int),
        metrics (dict): {
            "READ":  {"ops": float, "avg_us": float, "p99_us": float or None},
            "TOTAL": {"ops": float, "avg_us": float, "p99_us": float or None},
            "UPDATE": {...}
        }
    """
    threadcount = None
    metrics = {
        "READ":   {"ops": None, "avg_us": None, "p99_us": None},
        "TOTAL":  {"ops": None, "avg_us": None, "p99_us": None},
        "UPDATE": {"ops": None, "avg_us": None, "p99_us": None},
    }

    with open(path, "r") as f:
        for line in f:
            if threadcount is None:
                m_tc = THREAD_RE.search(line)
                if m_tc:
                    threadcount = int(m_tc.group(1))

            m = LINE_RE.match(line)
            if m:
                op = m.group(1)        # READ / TOTAL / UPDATE
                ops = float(m.group(4))
                avg_us = float(m.group(5))

                m_p99 = P99_RE.search(line)
                p99_us = float(m_p99.group(1)) if m_p99 else None

                metrics[op] = {"ops": ops, "avg_us": avg_us, "p99_us": p99_us}

    if threadcount is None:
        raise ValueError(f"Could not find threadcount in {path}")
    if metrics["TOTAL"]["ops"] is None:
        raise ValueError(f"Could not find TOTAL metrics in {path}")

    return threadcount, metrics


def infer_cache_name(path):
    """
    Infer cache name from filename.

    Expected patterns, e.g.:
      MultithreadScaling_wlA_lfu_zipfian_t1_ASYNC.out   -> 'lfu'
      MultithreadScaling_wlA_detox_zipfian_t8_ASYNC.out -> 'detox'
    """
    base = os.path.basename(path)
    name_no_ext = os.path.splitext(base)[0]
    parts = name_no_ext.split("_")

    # MultithreadScaling | wlA | <cache> | ...
    if len(parts) >= 3:
        return parts[2].lower()

    if len(parts) >= 1:
        return parts[-1].lower()

    return "cache"


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_all_caches_scaling.py <log1> <log2> ...")
        sys.exit(1)

    files = sys.argv[1:]

    # results[cache][threadcount] = {"ops": float, "p99": float}
    results = {}

    for path in files:
        try:
            tc, metrics = parse_log(path)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        cache = infer_cache_name(path)
        ops = metrics["TOTAL"]["ops"]
        p99 = metrics["TOTAL"]["p99_us"]

        if ops is None:
            print(f"[WARN] No TOTAL OPS in {path}, skipping.")
            continue
        if p99 is None:
            print(f"[WARN] No p99 TOTAL latency in {path}, skipping p99 for this file.")

        if cache not in results:
            results[cache] = {}
        # last file for a given (cache, threadcount) wins
        results[cache][tc] = {"ops": ops, "p99": p99}

    if not results:
        print("No valid data parsed. Exiting.")
        sys.exit(1)

    # Collect unified set of threadcounts across all caches
    all_threads = sorted({tc for cache in results.values() for tc in cache.keys()})

    # -------- Plot p99 latency --------
    fig_p99, ax_p99 = plt.subplots()

    for cache, tc_map in sorted(results.items()):
        tcs = sorted(tc_map.keys())
        p99_vals = [tc_map[t]["p99"] for t in tcs]

        ax_p99.plot(tcs, p99_vals, marker="o", label=cache.upper())

    ax_p99.set_xlabel("Thread count")
    ax_p99.set_ylabel("p99 TOTAL latency (µs)")
    ax_p99.set_ylim(*P99_LATENCY_YLIM)

    # x-ticks: standard set intersected with what we actually have
    standard_ticks = [1, 8, 16, 32, 64, 128]
    ticks = [t for t in standard_ticks if t in all_threads] or all_threads
    ax_p99.set_xticks(ticks)

    ax_p99.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)
    ax_p99.legend(loc="best")

    plt.title("Multithread scaling – p99 TOTAL latency vs thread count")
    fig_p99.tight_layout()

    out_p99 = "multithread_scaling_all_caches_p99.png"
    plt.savefig(out_p99, dpi=150)
    print(f"Saved combined p99 plot to {out_p99}")

    # -------- Plot throughput --------
    fig_tp, ax_tp = plt.subplots()

    for cache, tc_map in sorted(results.items()):
        tcs = sorted(tc_map.keys())
        ops_vals = [tc_map[t]["ops"] for t in tcs]

        ax_tp.plot(tcs, ops_vals, marker="o", label=cache.upper())

    ax_tp.set_xlabel("Thread count")
    ax_tp.set_ylabel("Throughput (ops/sec)")
    ax_tp.set_ylim(*THROUGHPUT_YLIM)

    ticks = [t for t in standard_ticks if t in all_threads] or all_threads
    ax_tp.set_xticks(ticks)

    ax_tp.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)
    ax_tp.legend(loc="best")

    plt.title("Multithread scaling – TOTAL throughput vs thread count")
    fig_tp.tight_layout()

    out_tp = "multithread_scaling_all_caches_throughput.png"
    plt.savefig(out_tp, dpi=150)
    print(f"Saved combined throughput plot to {out_tp}")


if __name__ == "__main__":
    main()
