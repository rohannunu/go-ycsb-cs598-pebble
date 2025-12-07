import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CACHE_NAMES = {"none", "lru", "lfu", "density", "detox"}
DIST_NAMES = {"zipfian", "uniform", "hotspot", "sequential"}


def parse_filename_metadata(path):
    """
    Expect patterns like:
      WorkloadVsDistribution_wlA_lru_zipfian_t32_ASYNC.out

    Extract:
      workload = 'A'
      cache = 'lru'
      distribution = 'zipfian'
      threadcount = 32
    """
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")

    workload = None
    cache = None
    distribution = None
    threadcount = None

    for p in parts:
        pl = p.lower()

        # workload part like "wlA" or "wla"
        if pl.startswith("wl") and len(pl) >= 3:
            workload = pl[2].upper()

        # cache name: none, lru, lfu, density, detox
        if pl in CACHE_NAMES:
            cache = pl

        # distribution: zipfian, uniform, hotspot, sequential
        if pl in DIST_NAMES:
            distribution = pl

        # thread count like "t32"
        m = re.fullmatch(r"t(\d+)", pl)
        if m:
            threadcount = int(m.group(1))

    return workload, cache, distribution, threadcount


def parse_log_file(path):
    with open(path, "r") as f:
        text = f.read()

    # First, infer from filename
    workload, cache, distribution, threadcount = parse_filename_metadata(path)

    # Fallbacks / cross-check from contents if needed
    if distribution is None:
        m = re.search(r'"requestdistribution"="([^"]+)"', text)
        if m:
            distribution = m.group(1).lower()

    if threadcount is None:
        m = re.search(r'"threadcount"="([^"]+)"', text)
        if m:
            try:
                threadcount = int(m.group(1))
            except ValueError:
                threadcount = None

    if cache is None:
        m = re.search(r'registering pebble\s+([A-Za-z0-9]+)\s+wrapper', text)
        if m:
            cache = m.group(1).lower()

    # Metrics to parse from final stats
    total_ops = None
    total_p99_us = None
    read_p99_us = None
    hit_rate = None
    repeat_ratio = None

    # TOTAL line: last occurrence
    total_pattern = re.compile(
        r'^TOTAL\s+-.*OPS:\s*([0-9.]+).*99th\(us\):\s*([0-9.]+)',
        re.M,
    )
    for m in total_pattern.finditer(text):
        total_ops = float(m.group(1))
        total_p99_us = float(m.group(2))

    # READ line p99 (optional)
    read_pattern = re.compile(r'^READ\s+-.*99th\(us\):\s*([0-9.]+)', re.M)
    for m in read_pattern.finditer(text):
        read_p99_us = float(m.group(1))

    # Hit rate
    m = re.search(r'Hit Rate:\s*([0-9.]+)%', text)
    if m:
        hit_rate = float(m.group(1)) / 100.0

    # Repeat ratio
    m = re.search(r'Repeat ratio:\s*([0-9.]+)%?', text)
    if m:
        repeat_ratio = float(m.group(1)) / 100.0

    return {
        "file": os.path.basename(path),
        "workload": workload or "UNKNOWN",
        "cache": cache,
        "distribution": distribution,
        "threadcount": threadcount,
        "total_ops": total_ops,
        "total_p99_us": total_p99_us,
        "read_p99_us": read_p99_us,
        "hit_rate": hit_rate,
        "repeat_ratio": repeat_ratio,
    }


def compute_speedups(df):
    """
    Add 'speedup' column: total_ops(cache) / total_ops(no-cache)
    Baseline assumed cache in {'none', 'nocache', 'baseline'}.
    """
    df = df.copy()
    df["cache_norm"] = df["cache"].fillna("unknown").str.lower()

    baseline_names = {"none", "nocache", "baseline"}
    baseline_df = df[df["cache_norm"].isin(baseline_names)]

    baseline_map = (
        baseline_df
        .set_index(["workload", "distribution"])["total_ops"]
        .to_dict()
    )

    def get_speedup(row):
        key = (row["workload"], row["distribution"])
        base_ops = baseline_map.get(key)
        if base_ops is None or base_ops == 0 or row["total_ops"] is None:
            return np.nan
        return row["total_ops"] / base_ops

    df["speedup"] = df.apply(get_speedup, axis=1)
    return df


def compute_latency_speedups(df):
    """
    Add 'latency_speedup' column:
      latency_speedup = p99_latency(no-cache) / p99_latency(cache)
    Using READ p99 latency. >1 => cache has lower p99 latency than baseline.
    """
    df = df.copy()
    df["cache_norm"] = df["cache"].fillna("unknown").str.lower()

    baseline_names = {"none", "nocache", "baseline"}
    baseline_df = df[df["cache_norm"].isin(baseline_names)]

    baseline_map = (
        baseline_df
        .set_index(["workload", "distribution"])["read_p99_us"]
        .to_dict()
    )

    def get_lat_speedup(row):
        key = (row["workload"], row["distribution"])
        base_lat = baseline_map.get(key)
        if base_lat is None or base_lat == 0 or row["read_p99_us"] is None:
            return np.nan
        return base_lat / row["read_p99_us"]

    df["latency_speedup"] = df.apply(get_lat_speedup, axis=1)
    return df


def plot_heatmap_cache_vs_workload(df, outdir):
    data = (
        df
        .dropna(subset=["speedup"])
        .groupby(["cache", "workload"])["speedup"]
        .mean()
        .reset_index()
    )

    if data.empty:
        print("No data for heatmap (cache vs workload).")
        return

    pivot = data.pivot(index="cache", columns="workload", values="speedup")

    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Avg speedup vs no-cache")

    caches = list(pivot.index)
    workloads = list(pivot.columns)

    plt.xticks(np.arange(len(workloads)), workloads)
    plt.yticks(np.arange(len(caches)), caches)

    # Add numeric labels inside each cell
    for i, cache in enumerate(caches):
        for j, wl in enumerate(workloads):
            val = pivot.loc[cache, wl]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Cache vs Workload (throughput speedup)")
    plt.xlabel("Workload")
    plt.ylabel("Cache")
    plt.tight_layout()
    out_path = os.path.join(outdir, "heatmap_cache_vs_workload.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_heatmap_cache_vs_distribution(df, outdir):
    """
    Heatmap of avg throughput speedup for each (cache, distribution).
    """
    data = (
        df
        .dropna(subset=["speedup"])
        .groupby(["cache", "distribution"])["speedup"]
        .mean()
        .reset_index()
    )

    if data.empty:
        print("No data for heatmap (cache vs distribution).")
        return

    pivot = data.pivot(index="cache", columns="distribution", values="speedup")

    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Avg speedup vs no-cache")

    caches = list(pivot.index)
    dists = list(pivot.columns)

    plt.xticks(np.arange(len(dists)), dists, rotation=15)
    plt.yticks(np.arange(len(caches)), caches)

    # Add numeric labels inside each cell
    for i, cache in enumerate(caches):
        for j, dist in enumerate(dists):
            val = pivot.loc[cache, dist]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Cache vs Distribution (throughput speedup)")
    plt.xlabel("Distribution")
    plt.ylabel("Cache")
    plt.tight_layout()
    out_path = os.path.join(outdir, "heatmap_cache_vs_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_latency_heatmap_cache_vs_workload(df, outdir):
    data = (
        df
        .dropna(subset=["latency_speedup"])
        .groupby(["cache", "workload"])["latency_speedup"]
        .mean()
        .reset_index()
    )

    if data.empty:
        print("No data for latency heatmap (cache vs workload).")
        return

    pivot = data.pivot(index="cache", columns="workload", values="latency_speedup")

    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Avg p99 latency speedup vs no-cache")

    caches = list(pivot.index)
    workloads = list(pivot.columns)

    plt.xticks(np.arange(len(workloads)), workloads)
    plt.yticks(np.arange(len(caches)), caches)

    for i, cache in enumerate(caches):
        for j, wl in enumerate(workloads):
            val = pivot.loc[cache, wl]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Cache vs Workload (p99 latency speedup)")
    plt.xlabel("Workload")
    plt.ylabel("Cache")
    plt.tight_layout()
    out_path = os.path.join(outdir, "latency_heatmap_cache_vs_workload.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_latency_heatmap_cache_vs_distribution(df, outdir):
    data = (
        df
        .dropna(subset=["latency_speedup"])
        .groupby(["cache", "distribution"])["latency_speedup"]
        .mean()
        .reset_index()
    )

    if data.empty:
        print("No data for latency heatmap (cache vs distribution).")
        return

    pivot = data.pivot(index="cache", columns="distribution", values="latency_speedup")

    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Avg p99 latency speedup vs no-cache")

    caches = list(pivot.index)
    dists = list(pivot.columns)

    plt.xticks(np.arange(len(dists)), dists, rotation=15)
    plt.yticks(np.arange(len(caches)), caches)

    for i, cache in enumerate(caches):
        for j, dist in enumerate(dists):
            val = pivot.loc[cache, dist]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Cache vs Distribution (p99 latency speedup)")
    plt.xlabel("Distribution")
    plt.ylabel("Cache")
    plt.tight_layout()
    out_path = os.path.join(outdir, "latency_heatmap_cache_vs_distribution.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_overall_cache_speedup(df, outdir):
    data = (
        df
        .dropna(subset=["speedup"])
        .groupby("cache")["speedup"]
        .mean()
        .sort_values(ascending=False)
    )

    if data.empty:
        print("No data for overall cache speedup.")
        return

    plt.figure(figsize=(6, 4))
    bars = plt.bar(data.index, data.values)
    plt.ylabel("Avg speedup vs no-cache")
    plt.title("Overall Cache Performance (throughput)")

    # Labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = os.path.join(outdir, "overall_cache_speedup.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_win_counts(df, outdir):
    group_cols = ["workload", "distribution"]
    df_valid = df.dropna(subset=["total_ops", "cache"])

    if df_valid.empty:
        print("No valid data for win counts.")
        return

    idx = df_valid.groupby(group_cols)["total_ops"].idxmax()
    winners = df_valid.loc[idx]

    win_counts = winners.groupby("cache").size().sort_values(ascending=False)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(win_counts.index, win_counts.values)
    plt.ylabel("# of (workload, dist) wins")
    plt.title("Win Count per Cache (by throughput)")

    # Labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = os.path.join(outdir, "cache_win_counts.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_distribution_improvement(df, outdir):
    df = df.copy()
    df["cache_norm"] = df["cache"].fillna("unknown").str.lower()
    baseline_names = {"none", "nocache", "baseline"}

    caches_only = df[
        (~df["cache_norm"].isin(baseline_names))
        & (~df["speedup"].isna())
    ]

    if caches_only.empty:
        print("No data for distribution improvement.")
        return

    dist_improv = (
        caches_only
        .groupby("distribution")["speedup"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(6, 4))
    bars = plt.bar(dist_improv.index, dist_improv.values)
    plt.ylabel("Avg speedup vs no-cache")
    plt.title("Cache Benefit by Distribution (throughput)")

    # Labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = os.path.join(outdir, "distribution_improvement.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_best_cache_per_distribution(df, outdir):
    """
    For each distribution, find the cache (including baseline 'none')
    with the highest average speedup vs no-cache, and plot it.
    """
    df_valid = df.dropna(subset=["speedup", "cache", "distribution"])

    if df_valid.empty:
        print("No valid data for best-cache-per-distribution plot.")
        return

    # Average speedup per (distribution, cache)
    perf = (
        df_valid
        .groupby(["distribution", "cache"])["speedup"]
        .mean()
        .reset_index()
    )

    # For each distribution, pick the cache with max avg speedup
    idx = perf.groupby("distribution")["speedup"].idxmax()
    best = perf.loc[idx].sort_values("distribution")

    # Save table as CSV
    out_csv = os.path.join(outdir, "best_cache_per_distribution.csv")
    best.to_csv(out_csv, index=False)
    print(f"Saved best-per-distribution table to {out_csv}")

    # Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar(best["distribution"], best["speedup"])
    plt.ylabel("Best avg speedup vs no-cache")
    plt.title("Best Cache per Distribution (throughput)")

    for bar, (_, row) in zip(bars, best.iterrows()):
        height = bar.get_height()
        label = f"{row['cache']} ({height:.2f}x)"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_png = os.path.join(outdir, "best_cache_per_distribution.png")
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")


def plot_best_cache_per_workload(df, outdir):
    """
    For each workload (A, B, C...), find the cache (including 'none')
    with the highest average speedup vs no-cache, and plot it.
    """
    df_valid = df.dropna(subset=["speedup", "cache", "workload"])

    if df_valid.empty:
        print("No valid data for best-cache-per-workload plot.")
        return

    # Average speedup per (workload, cache)
    perf = (
        df_valid
        .groupby(["workload", "cache"])["speedup"]
        .mean()
        .reset_index()
    )

    # For each workload, pick the cache with max avg speedup
    idx = perf.groupby("workload")["speedup"].idxmax()
    best = perf.loc[idx].sort_values("workload")

    # Save table as CSV
    out_csv = os.path.join(outdir, "best_cache_per_workload.csv")
    best.to_csv(out_csv, index=False)
    print(f"Saved best-per-workload table to {out_csv}")

    # Plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar(best["workload"], best["speedup"])
    plt.ylabel("Best avg speedup vs no-cache")
    plt.title("Best Cache per Workload (throughput)")

    for bar, (_, row) in zip(bars, best.iterrows()):
        height = bar.get_height()
        label = f"{row['cache']} ({height:.2f}x)"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_png = os.path.join(outdir, "best_cache_per_workload.png")
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Analyze YCSB logs for cache experiments.")
    parser.add_argument("log_dir", help="Directory containing YCSB log files")
    parser.add_argument("--outdir", default="plots", help="Directory to save plots")
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".log", ".txt", ".out"],
        help="File extensions to treat as logs",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for root, _, files in os.walk(args.log_dir):
        for name in files:
            if any(name.endswith(ext) for ext in args.exts):
                path = os.path.join(root, name)
                record = parse_log_file(path)
                if record["total_ops"] is None:
                    print(f"Skipping {name}: could not parse TOTAL OPS.")
                    continue
                rows.append(record)

    if not rows:
        print("No valid log files found.")
        return

    df = pd.DataFrame(rows)
    df = compute_speedups(df)
    df = compute_latency_speedups(df)

    summary_path = os.path.join(args.outdir, "ycsb_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"Wrote summary CSV to {summary_path}")

    # Throughput-based plots
    plot_heatmap_cache_vs_workload(df, args.outdir)
    plot_heatmap_cache_vs_distribution(df, args.outdir)
    plot_overall_cache_speedup(df, args.outdir)
    plot_win_counts(df, args.outdir)
    plot_distribution_improvement(df, args.outdir)
    plot_best_cache_per_distribution(df, args.outdir)
    plot_best_cache_per_workload(df, args.outdir)

    # Latency-based plots (p99 READ)
    plot_latency_heatmap_cache_vs_workload(df, args.outdir)
    plot_latency_heatmap_cache_vs_distribution(df, args.outdir)


if __name__ == "__main__":
    main()
