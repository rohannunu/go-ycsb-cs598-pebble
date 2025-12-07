#!/usr/bin/env python3
import re
import sys
import os

import matplotlib
matplotlib.use("Agg")  # non-GUI backend so we can save figures on a server
import matplotlib.pyplot as plt

# =======================
# Config: shared y-limits
# =======================
# Global ranges across all caches
THROUGHPUT_YLIM = (0, 100000)   # ops/sec
AVG_LATENCY_YLIM = (0, 7000)    # µs
P99_LATENCY_YLIM = (0, 20000)   # µs (tail can be much higher)

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

                # Try to get p99 from the same line
                m_p99 = P99_RE.search(line)
                p99_us = float(m_p99.group(1)) if m_p99 else None

                # last occurrence wins → final snapshot
                metrics[op] = {"ops": ops, "avg_us": avg_us, "p99_us": p99_us}

    if threadcount is None:
        raise ValueError(f"Could not find threadcount in {path}")
    if metrics["TOTAL"]["ops"] is None:
        raise ValueError(f"Could not find TOTAL metrics in {path}")
    if metrics["READ"]["ops"] is None:
        raise ValueError(f"Could not find READ metrics in {path}")

    return threadcount, metrics


def infer_cache_name(first_file):
    """
    Infer cache name from first filename.

    Expected patterns, e.g.:
      MultithreadScaling_wlA_lfu_zipfian_t1_ASYNC.out   -> 'lfu'
      MultithreadScaling_wlA_detox_zipfian_t8_ASYNC.out -> 'detox'

    i.e., use the 3rd underscore-separated token.
    """
    base = os.path.basename(first_file)
    name_no_ext = os.path.splitext(base)[0]  # strip .out
    parts = name_no_ext.split("_")

    # Prefer the 3rd token: MultithreadScaling | wlA | <cache> | ...
    if len(parts) >= 3:
        return parts[2].lower()

    # Fallback: last token if the pattern is weird
    if len(parts) >= 1:
        return parts[-1].lower()

    return "cache"


def plot_avg_latency_scaling(threadcounts, throughputs, avg_latencies, cache_name):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Thread count")
    ax1.set_ylabel("Throughput (ops/sec)")

    # Force x-axis ticks to typical thread counts you used
    tick_values = [1, 8, 16, 32, 64, 128]
    ax1.set_xticks(tick_values)

    # Apply shared y-limits for throughput
    ax1.set_ylim(*THROUGHPUT_YLIM)

    # Throughput: blue line
    ln1 = ax1.plot(
        threadcounts,
        throughputs,
        marker="o",
        color="tab:blue",
        label="Throughput (TOTAL OPS)"
    )
    ax1.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg TOTAL latency (µs)")

    # Apply shared y-limits for avg latency
    ax2.set_ylim(*AVG_LATENCY_YLIM)

    # Latency: orange line
    ln2 = ax2.plot(
        threadcounts,
        avg_latencies,
        marker="s",
        color="tab:orange",
        label="Avg TOTAL latency"
    )

    # Single combined legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title(f"{cache_name.upper()} cache scaling: throughput and avg latency vs thread count")
    fig.tight_layout()

    out_name = f"multithread_scaling_{cache_name}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved plot to {out_name}")


def plot_p99_latency_scaling(threadcounts, throughputs, p99_latencies, cache_name):
    # If any p99 is missing, skip the tail plot
    if any(lat is None for lat in p99_latencies):
        print("Some p99 latency values missing; skipping tail-latency scaling plot.")
        return

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Thread count")
    ax1.set_ylabel("Throughput (ops/sec)")

    tick_values = [1, 8, 16, 32, 64, 128]
    ax1.set_xticks(tick_values)
    ax1.set_ylim(*THROUGHPUT_YLIM)

    ln1 = ax1.plot(
        threadcounts,
        throughputs,
        marker="o",
        color="tab:blue",
        label="Throughput (TOTAL OPS)"
    )
    ax1.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("p99 TOTAL latency (µs)")
    ax2.set_ylim(*P99_LATENCY_YLIM)

    ln2 = ax2.plot(
        threadcounts,
        p99_latencies,
        marker="^",
        color="tab:red",
        label="p99 TOTAL latency"
    )

    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title(f"{cache_name.upper()} cache scaling: throughput and p99 latency vs thread count")
    fig.tight_layout()

    out_name = f"multithread_scaling_tail_{cache_name}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved plot to {out_name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_ycsb_scaling.py <log1> <log2> ...")
        sys.exit(1)

    files = sys.argv[1:]
    data = {}

    for path in files:
        try:
            tc, metrics = parse_log(path)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        if tc in data:
            print(f"[WARN] Duplicate threadcount {tc} (file {path}); overwriting.")

        throughput = metrics["TOTAL"]["ops"]
        avg_latency = metrics["TOTAL"]["avg_us"]
        p99_latency = metrics["TOTAL"]["p99_us"]

        data[tc] = {
            "throughput_ops": throughput,
            "avg_latency_us": avg_latency,
            "p99_latency_us": p99_latency,
        }

    if not data:
        print("No valid data parsed. Exiting.")
        sys.exit(1)

    # Sort by thread count
    threadcounts = sorted(data.keys())
    throughputs = [data[t]["throughput_ops"] for t in threadcounts]
    avg_latencies = [data[t]["avg_latency_us"] for t in threadcounts]
    p99_latencies = [data[t]["p99_latency_us"] for t in threadcounts]

    # Infer cache name from first file for output filename/title
    cache_name = infer_cache_name(files[0])

    # Plot avg latency + throughput
    plot_avg_latency_scaling(threadcounts, throughputs, avg_latencies, cache_name)

    # Plot p99 (tail) latency + throughput
    plot_p99_latency_scaling(threadcounts, throughputs, p99_latencies, cache_name)


if __name__ == "__main__":
    main()
