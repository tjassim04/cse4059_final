#!/usr/bin/env python3
"""
analyze_results.py

Generate figures, summary tables, and a markdown report from:
  - solver wall-clock results (results.log written by run_suite.sh)
  - Nsight Compute CSVs from profile_suite.sh

Usage:
    python3 analyze_results.py [profile_dir]
        [--results-log path/to/results.log]
        [--out-dir figures]

Dependencies: pandas, numpy, matplotlib.
    pip install pandas numpy matplotlib

Output structure (under --out-dir):
    fig01_speedup_grid.png         per-query speedup, one panel per dataset
    fig02_speedup_vs_size.png      median/max speedup vs |E|
    fig03_per_depth_time.png       stacked GPU time per depth, per query
    fig04_frontier_growth.png      frontier size vs depth (log scale)
    fig05_cpu_vs_gpu_scatter.png   CPU ms vs GPU ms (log-log)
    fig06_aggregate_speedup.png    aggregate speedup per dataset
    fig07_occupancy.png            achieved vs theoretical occupancy
    fig08_memory_throughput.png    DRAM/L1/L2 throughput per launch
    fig09_compute_vs_memory.png    SOL compute vs SOL memory scatter
    fig10_warp_states.png          stall reason breakdown
    fig11_sm_utilization.png       SM busy % distribution
    fig12_roofline.png             arithmetic intensity vs throughput
    summary_perf.csv               full per-query performance table
    summary_kernels.csv            full per-launch kernel metrics
    report.md                      markdown summary + key numbers
"""

from __future__ import annotations

import argparse
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_CPU = "#c0392b"
C_GPU = "#2980b9"
C_ACCENT = "#27ae60"
C_NEUTRAL = "#7f8c8d"


# ===========================================================================
# Solver log parser
# ===========================================================================

DATASET_HEADER_RE = re.compile(r"^={4,}\s+(\S+)\s+={4,}\s*$", re.MULTILINE)


def parse_results_log(text: str) -> Dict[str, dict]:
    """
    Parse a multi-dataset solver log. Returns:

      {
        "<dataset_name>": {
          "summary": DataFrame[name, cpu_matches, gpu_matches, cpu_ms, gpu_ms, speedup, ok],
          "queries": [
            {"name": ..., "per_depth_ms": [...], "gpu_frontiers": [...], "cpu_frontiers": [...]},
            ...
          ],
          "num_vertices": int or None,
          "num_edges": int or None,
        },
        ...
      }
    """
    # Find all dataset section boundaries.
    out = {}
    matches = list(DATASET_HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        # Skip headers that aren't dataset names.
        if not re.match(r"^\d+_", name):
            continue
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]

        # Per-dataset CSV summary.
        summary = pd.DataFrame()
        sm = re.search(
            r"^name,cpu_matches,gpu_matches,cpu_ms,gpu_ms,speedup,ok\s*\n(.*?)(?=\n=|\Z)",
            body, re.DOTALL | re.MULTILINE,
        )
        if sm:
            csv_text = (
                "name,cpu_matches,gpu_matches,cpu_ms,gpu_ms,speedup,ok\n"
                + sm.group(1).strip()
            )
            try:
                summary = pd.read_csv(StringIO(csv_text))
                summary["dataset"] = name
            except Exception as e:
                print(f"  warning: couldn't parse summary for {name}: {e}",
                      file=sys.stderr)

        # Per-query verbose info.
        queries = []
        for qm in re.finditer(
            r"=== Query:\s+(\S+)\s+===\s*\n(.*?)(?=\n=== Query:|\n=== Summary|\Z)",
            body, re.DOTALL,
        ):
            qname = qm.group(1).strip()
            qbody = qm.group(2)

            per_depth_ms = []
            pdm = re.search(r"GPU per-depth ms:\s+(.+)", qbody)
            if pdm:
                per_depth_ms = [float(x) for x in pdm.group(1).split()]

            gpu_frontiers = [
                int(s) for _, s in
                re.findall(r"\[GPU\] depth (\d+) frontier:\s+(\d+)", qbody)
            ]
            cpu_frontiers = [
                int(s) for _, s in
                re.findall(r"\[CPU\] depth (\d+) frontier:\s+(\d+)", qbody)
            ]

            queries.append({
                "name": qname,
                "per_depth_ms": per_depth_ms,
                "gpu_frontiers": gpu_frontiers,
                "cpu_frontiers": cpu_frontiers,
            })

        # Graph stats.
        v = e = None
        gm = re.search(r"data graph:\s+(\d+) vertices,\s+(\d+) logical edges", body)
        if gm:
            v, e = int(gm.group(1)), int(gm.group(2))

        out[name] = {
            "summary": summary,
            "queries": queries,
            "num_vertices": v,
            "num_edges": e,
        }
    return out


# ===========================================================================
# ncu CSV parser
# ===========================================================================

def parse_ncu_csv(path: Path) -> pd.DataFrame:
    """Parse one ncu CSV file. Returns a long-form DataFrame, possibly empty."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    text = path.read_text(errors="replace")
    lines = text.splitlines()

    # ncu often prepends a few lines before the CSV header. Find the header.
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID,"):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    body = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(StringIO(body), low_memory=False)
    except Exception as e:
        print(f"  warning: parse error in {path.name}: {e}", file=sys.stderr)
        return pd.DataFrame()

    df["__source"] = path.stem
    return df


def pivot_metrics_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-form ncu CSV (one row per metric) into wide form (one row
    per kernel launch with metrics as columns).
    """
    if df.empty or "Metric Name" not in df.columns:
        return df

    df = df.copy()

    def to_num(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().replace(",", "")
        try:
            return float(s)
        except ValueError:
            return np.nan

    df["__value"] = df["Metric Value"].apply(to_num)

    id_cols = [c for c in
               ["__source", "ID", "Kernel Name", "Block Size", "Grid Size"]
               if c in df.columns]

    try:
        wide = (df.pivot_table(
                    index=id_cols,
                    columns="Metric Name",
                    values="__value",
                    aggfunc="first")
                  .reset_index())
    except Exception as e:
        print(f"  warning: pivot failed: {e}", file=sys.stderr)
        return pd.DataFrame()

    return wide


def find_metric_column(df: pd.DataFrame, *patterns: str) -> Optional[str]:
    """First column whose name matches any of the given case-insensitive substrings."""
    if df.empty:
        return None
    cols = list(df.columns)
    for pat in patterns:
        pat_lower = pat.lower()
        for c in cols:
            if pat_lower in c.lower():
                return c
    return None


# ===========================================================================
# Performance figures (from results.log)
# ===========================================================================

def fig01_speedup_grid(datasets: Dict[str, dict], out: Path) -> None:
    """One panel per dataset showing speedup-per-query bars."""
    keys = sorted(k for k in datasets if not datasets[k]["summary"].empty)
    if not keys:
        return

    n = len(keys)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.6 * rows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        d = datasets[key]
        df = d["summary"]
        if df.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(df))
        bars = ax.bar(x, df["speedup"], color=C_GPU, edgecolor="black",
                      linewidth=0.5)

        # Annotate values (small font, vertical).
        for xi, sp in zip(x, df["speedup"]):
            ax.text(xi, sp * 1.05, f"{sp:.0f}", ha="center", va="bottom",
                    fontsize=7, rotation=0)

        v = d["num_vertices"]
        e = d["num_edges"]
        title = f"{key}"
        if v is not None and e is not None:
            title += f"  (V={v:,}, E={e:,})"
        ax.set_title(title, fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(df["name"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Speedup vs CPU (×)")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.axhline(1.0, color=C_NEUTRAL, linestyle="--", linewidth=1, alpha=0.5)

    for ax in axes[len(keys):]:
        ax.set_visible(False)

    fig.suptitle("Per-query GPU vs CPU speedup, by dataset", fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig02_speedup_vs_size(datasets: Dict[str, dict], out: Path) -> None:
    """Median/min/max speedup per dataset, plotted vs |E|."""
    rows = []
    for key, d in datasets.items():
        df = d["summary"]
        if df.empty or d["num_edges"] is None:
            continue
        # Filter out queries where CPU time is < 5ms (noise regime).
        df_clean = df[df["cpu_ms"] >= 5.0]
        if df_clean.empty:
            df_clean = df
        rows.append({
            "dataset": key,
            "num_vertices": d["num_vertices"],
            "num_edges": d["num_edges"],
            "median_speedup": df_clean["speedup"].median(),
            "max_speedup": df_clean["speedup"].max(),
            "min_speedup": df_clean["speedup"].min(),
        })
    if not rows:
        return
    sdf = pd.DataFrame(rows).sort_values("num_edges")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = sdf["num_edges"]
    ax.plot(x, sdf["median_speedup"], "o-", color=C_GPU, linewidth=2,
            markersize=8, label="median")
    ax.fill_between(x, sdf["min_speedup"], sdf["max_speedup"],
                    alpha=0.2, color=C_GPU, label="min–max range")

    for _, r in sdf.iterrows():
        ax.annotate(r["dataset"],
                    (r["num_edges"], r["median_speedup"]),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=8, color=C_NEUTRAL)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Graph edges  |E|")
    ax.set_ylabel("Speedup vs CPU (×)")
    ax.set_title("Speedup vs graph size  (queries with CPU time ≥ 5ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig03_per_depth_time(datasets: Dict[str, dict], out: Path) -> None:
    """For one mid-size dataset: stacked bar of per-depth GPU time."""
    target_keys = [k for k in datasets
                   if "03_er_n2k_d8" in k or "02_er_n1k_d6" in k]
    if not target_keys:
        target_keys = sorted(datasets.keys())[:1]
    if not target_keys:
        return
    key = target_keys[0]
    d = datasets[key]
    queries = d["queries"]
    if not queries:
        return

    max_depth = max(len(q["per_depth_ms"]) for q in queries) or 0
    if max_depth == 0:
        return

    names = [q["name"] for q in queries]
    arr = np.zeros((len(queries), max_depth))
    for i, q in enumerate(queries):
        for j, t in enumerate(q["per_depth_ms"]):
            arr[i, j] = t

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    cmap = plt.get_cmap("viridis", max_depth)

    for d_idx in range(max_depth):
        vals = arr[:, d_idx]
        ax.bar(x, vals, bottom=bottom, color=cmap(d_idx),
               label=f"depth {d_idx + 1}", edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("GPU time (ms)")
    ax.set_yscale("log")
    ax.set_title(f"GPU time per depth — {key}")
    ax.legend(ncol=max_depth, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig04_frontier_growth(datasets: Dict[str, dict], out: Path) -> None:
    """Frontier size vs depth, one line per query, for one mid-size dataset."""
    target_keys = [k for k in datasets
                   if "03_er_n2k_d8" in k or "02_er_n1k_d6" in k]
    if not target_keys:
        target_keys = sorted(datasets.keys())[:1]
    if not target_keys:
        return
    key = target_keys[0]
    queries = datasets[key]["queries"]
    if not queries:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab20", len(queries))
    for i, q in enumerate(queries):
        fr = q["gpu_frontiers"]
        if not fr:
            continue
        depths = list(range(1, len(fr) + 1))
        ax.plot(depths, fr, "o-", color=cmap(i), label=q["name"],
                alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Frontier size  (# partial mappings)")
    ax.set_yscale("log")
    ax.set_title(f"Frontier size vs depth — {key}")
    ax.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig05_cpu_vs_gpu_scatter(datasets: Dict[str, dict], out: Path) -> None:
    """Log-log scatter of CPU ms vs GPU ms across all queries and datasets."""
    pts = []
    for key, d in datasets.items():
        df = d["summary"]
        if df.empty:
            continue
        for _, r in df.iterrows():
            pts.append((r["cpu_ms"], r["gpu_ms"], key, r["name"]))
    if not pts:
        return
    df = pd.DataFrame(pts, columns=["cpu_ms", "gpu_ms", "dataset", "query"])

    fig, ax = plt.subplots(figsize=(8, 8))
    keys = sorted(df["dataset"].unique())
    cmap = plt.get_cmap("tab10", len(keys))
    for i, k in enumerate(keys):
        sub = df[df["dataset"] == k]
        ax.scatter(sub["cpu_ms"], sub["gpu_ms"], s=40, alpha=0.7,
                   color=cmap(i), label=k, edgecolor="black", linewidth=0.4)

    # Reference lines: 1×, 100×, 1000× speedup.
    lo = max(1e-3, df[["cpu_ms", "gpu_ms"]].min().min())
    hi = df[["cpu_ms", "gpu_ms"]].max().max()
    ref_x = np.geomspace(lo, hi, 50)
    for factor, label in [(1, "1×"), (100, "100×"), (1000, "1000×")]:
        ax.plot(ref_x, ref_x / factor, color=C_NEUTRAL, linestyle="--",
                linewidth=0.8, alpha=0.6)
        ax.text(hi, hi / factor, label, color=C_NEUTRAL, fontsize=8,
                ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU time (ms)")
    ax.set_ylabel("GPU time (ms)")
    ax.set_title("CPU vs GPU time per query  (lower-right = bigger speedup)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig06_aggregate_speedup(datasets: Dict[str, dict], out: Path) -> None:
    """Aggregate speedup per dataset (sum CPU / sum GPU)."""
    rows = []
    for key, d in datasets.items():
        df = d["summary"]
        if df.empty:
            continue
        cpu_total = df["cpu_ms"].sum()
        gpu_total = df["gpu_ms"].sum()
        if gpu_total <= 0:
            continue
        rows.append({
            "dataset": key,
            "cpu_ms": cpu_total,
            "gpu_ms": gpu_total,
            "speedup": cpu_total / gpu_total,
            "num_edges": d["num_edges"] or 0,
        })
    if not rows:
        return
    sdf = pd.DataFrame(rows).sort_values("num_edges")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sdf))
    ax.bar(x, sdf["speedup"], color=C_ACCENT, edgecolor="black", linewidth=0.5)
    for xi, sp in zip(x, sdf["speedup"]):
        ax.text(xi, sp * 1.02, f"{sp:.1f}×", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sdf["dataset"], rotation=20, ha="right")
    ax.set_ylabel("Aggregate speedup  (Σ CPU / Σ GPU)")
    ax.set_title("Aggregate GPU vs CPU speedup, per dataset")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ===========================================================================
# Profiling figures (from ncu CSVs)
# ===========================================================================

def fig07_occupancy(wide: pd.DataFrame, out: Path) -> None:
    if wide.empty:
        return
    achieved_col = find_metric_column(wide,
        "achieved active warps", "achieved_active",
        "sm__warps_active.avg.pct_of_peak", "achieved occupancy")
    theoretical_col = find_metric_column(wide,
        "theoretical occupancy", "theoretical active warps",
        "max active warps")
    if achieved_col is None:
        return

    vals = pd.to_numeric(wide[achieved_col], errors="coerce").dropna()
    if vals.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(vals, bins=20, color=C_GPU, edgecolor="black", linewidth=0.5)
    ax1.axvline(vals.mean(), color=C_CPU, linestyle="--", linewidth=2,
                label=f"mean = {vals.mean():.1f}")
    ax1.axvline(vals.median(), color=C_ACCENT, linestyle="--", linewidth=2,
                label=f"median = {vals.median():.1f}")
    ax1.set_xlabel(achieved_col)
    ax1.set_ylabel("Kernel launches")
    ax1.set_title("Distribution of achieved occupancy")
    ax1.legend()

    ax2.scatter(np.arange(len(vals)), vals.values, color=C_GPU, alpha=0.6,
                s=20)
    if theoretical_col is not None:
        tvals = pd.to_numeric(wide[theoretical_col], errors="coerce").dropna()
        if not tvals.empty:
            ax2.axhline(tvals.mean(), color=C_NEUTRAL, linestyle=":",
                        label=f"theoretical = {tvals.mean():.1f}")
    ax2.set_xlabel("Kernel launch index (chronological)")
    ax2.set_ylabel(achieved_col)
    ax2.set_title("Occupancy per launch")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig08_memory_throughput(wide: pd.DataFrame, out: Path) -> None:
    if wide.empty:
        return
    dram_col = find_metric_column(wide,
        "dram__bytes.sum.per_second", "dram throughput", "memory throughput")
    l1_col = find_metric_column(wide,
        "l1tex__t_bytes.sum.per_second", "l1/tex hit rate", "l1tex throughput")
    l2_col = find_metric_column(wide,
        "lts__t_bytes.sum.per_second", "l2 throughput", "l2 cache throughput")

    metrics = []
    for label, col in [("DRAM", dram_col), ("L1/TEX", l1_col), ("L2", l2_col)]:
        if col is not None:
            v = pd.to_numeric(wide[col], errors="coerce").dropna()
            if not v.empty:
                metrics.append((label, col, v))
    if not metrics:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5),
                             squeeze=False)
    for ax, (label, col, v) in zip(axes[0], metrics):
        ax.hist(v, bins=20, color=C_GPU, edgecolor="black", linewidth=0.5)
        ax.axvline(v.mean(), color=C_CPU, linestyle="--", linewidth=2,
                   label=f"mean = {v.mean():.2f}")
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Launches")
        ax.set_title(f"{label} throughput")
        ax.legend(fontsize=8)
    fig.suptitle("Memory subsystem throughput per kernel launch", fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig09_compute_vs_memory(wide: pd.DataFrame, out: Path) -> None:
    if wide.empty:
        return
    sm_col = find_metric_column(wide,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "compute (sm) throughput", "sm throughput")
    mem_col = find_metric_column(wide,
        "gpu__compute_memory_throughput", "memory throughput",
        "dram__throughput.avg.pct_of_peak", "memory (dram) throughput")
    if sm_col is None or mem_col is None:
        return

    sm = pd.to_numeric(wide[sm_col], errors="coerce")
    mem = pd.to_numeric(wide[mem_col], errors="coerce")
    valid = sm.notna() & mem.notna()
    if not valid.any():
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(mem[valid], sm[valid], s=40, alpha=0.6, color=C_GPU,
               edgecolor="black", linewidth=0.5)
    ax.plot([0, 100], [0, 100], color=C_NEUTRAL, linestyle="--",
            linewidth=1, label="balanced (compute = memory)")
    ax.set_xlim(0, max(100, mem[valid].max() * 1.05))
    ax.set_ylim(0, max(100, sm[valid].max() * 1.05))
    ax.set_xlabel(f"{mem_col}\n(memory utilization, % of peak)")
    ax.set_ylabel(f"{sm_col}\n(compute utilization, % of peak)")
    ax.set_title("Compute vs memory utilization\n"
                 "below the diagonal = memory-bound; above = compute-bound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig10_warp_states(wide: pd.DataFrame, out: Path) -> None:
    if wide.empty:
        return
    state_cols = [c for c in wide.columns
                  if "warp_cycles_per_issued_instruction" in c.lower()
                  or "smsp__average_warp_latency_per_inst" in c.lower()
                  or "stall" in c.lower()]
    if not state_cols:
        # Fallback: anything with 'warp' in it.
        state_cols = [c for c in wide.columns if "warp" in c.lower()
                      and "stall" in c.lower()]
    if not state_cols:
        return

    means = []
    for c in state_cols:
        v = pd.to_numeric(wide[c], errors="coerce").dropna()
        if not v.empty:
            means.append((c, v.mean()))
    if not means:
        return

    means.sort(key=lambda kv: kv[1], reverse=True)
    top = means[:12]

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))
    labels = [t[0] for t in top]
    vals = [t[1] for t in top]
    short = [
        re.sub(r"^.*?stall_", "stall_",
               re.sub(r"_per_warp_active.*$", "", l))[:60]
        for l in labels
    ]
    ax.barh(np.arange(len(top)), vals, color=C_GPU, edgecolor="black",
            linewidth=0.5)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(short, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Cycles per instruction (avg across launches)")
    ax.set_title("Top warp stall reasons")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig11_sm_utilization(wide: pd.DataFrame, out: Path) -> None:
    if wide.empty:
        return
    sm_col = find_metric_column(wide,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "compute (sm) throughput")
    if sm_col is None:
        return
    v = pd.to_numeric(wide[sm_col], errors="coerce").dropna()
    if v.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(v, bins=20, color=C_GPU, edgecolor="black", linewidth=0.5)
    ax.axvline(v.mean(), color=C_CPU, linestyle="--", linewidth=2,
               label=f"mean = {v.mean():.1f}%")
    ax.set_xlabel(f"{sm_col} (% of peak)")
    ax.set_ylabel("Kernel launches")
    ax.set_title("SM utilization distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig12_roofline(wide: pd.DataFrame, out: Path) -> None:
    if wide.empty:
        return
    # Try to find arithmetic intensity / throughput-related metrics.
    flops_col = find_metric_column(wide,
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_second",
        "achieved flops", "fp32 throughput")
    bytes_col = find_metric_column(wide,
        "dram__bytes.sum.per_second", "dram throughput")
    inst_col = find_metric_column(wide,
        "smsp__inst_executed.sum", "instructions executed")
    elapsed_col = find_metric_column(wide,
        "gpc__cycles_elapsed.max", "duration", "elapsed")

    # If we have neither flops nor instructions, give up gracefully.
    if (flops_col is None and inst_col is None) or bytes_col is None:
        # Make a placeholder so the report still has fig12 to reference.
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5,
                "Roofline metrics not present in profile.\n"
                "Required: DRAM throughput + (FLOPS or instructions/cycle).\n"
                "Run profile_suite.sh phase E with newer ncu.",
                ha="center", va="center", fontsize=11,
                transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(out)
        plt.close(fig)
        return

    bytes_per_sec = pd.to_numeric(wide[bytes_col], errors="coerce")
    if flops_col is not None:
        ops = pd.to_numeric(wide[flops_col], errors="coerce")
        ylabel = f"{flops_col}"
    else:
        ops = pd.to_numeric(wide[inst_col], errors="coerce")
        ylabel = f"{inst_col} (proxy for ops)"

    valid = bytes_per_sec.notna() & ops.notna() & (bytes_per_sec > 0)
    if not valid.any():
        return

    ai = ops[valid] / bytes_per_sec[valid]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(ai, ops[valid], s=40, alpha=0.7, color=C_GPU,
               edgecolor="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity  (ops / byte)")
    ax.set_ylabel(ylabel)
    ax.set_title("Roofline-style scatter\n"
                 "left of knee = memory-bound; right = compute-bound")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ===========================================================================
# Markdown report
# ===========================================================================

def write_report(datasets: Dict[str, dict], wide: pd.DataFrame,
                 out: Path, fig_dir: Path) -> None:
    """Write a markdown report summarizing perf and profiling."""
    lines: List[str] = []
    lines.append("# GPU Subgraph Matching: Performance & Profiling Report\n")
    lines.append(f"_Generated by `analyze_results.py`._\n")

    # ---- High-level numbers ----
    all_pts = []
    for key, d in datasets.items():
        df = d["summary"]
        if df.empty:
            continue
        for _, r in df.iterrows():
            all_pts.append({
                "dataset": key, "query": r["name"],
                "cpu_ms": r["cpu_ms"], "gpu_ms": r["gpu_ms"],
                "speedup": r["speedup"], "ok": r.get("ok", "PASS"),
            })
    perf_df = pd.DataFrame(all_pts)

    if not perf_df.empty:
        meaningful = perf_df[perf_df["cpu_ms"] >= 5.0]
        cpu_total = perf_df["cpu_ms"].sum()
        gpu_total = perf_df["gpu_ms"].sum()
        agg_speedup = (cpu_total / gpu_total) if gpu_total > 0 else float("nan")

        lines.append("## Headline numbers\n")
        lines.append(f"- Datasets profiled: **{perf_df['dataset'].nunique()}**")
        lines.append(f"- Total query runs: **{len(perf_df)}**")
        lines.append(f"- Correctness: **{(perf_df['ok'] == 'PASS').sum()} PASS"
                     f" / {len(perf_df)}**")
        lines.append(f"- Aggregate CPU time: **{cpu_total / 1000:,.2f} s**")
        lines.append(f"- Aggregate GPU time: **{gpu_total / 1000:,.2f} s**")
        lines.append(f"- Aggregate speedup (Σ CPU / Σ GPU): "
                     f"**{agg_speedup:,.1f}×**")
        if not meaningful.empty:
            lines.append(f"- Median per-query speedup (CPU ≥ 5 ms): "
                         f"**{meaningful['speedup'].median():,.1f}×**")
            top = meaningful.nlargest(1, "speedup").iloc[0]
            lines.append(f"- Largest speedup: **{top['speedup']:,.1f}×** "
                         f"({top['query']} on {top['dataset']}, "
                         f"CPU {top['cpu_ms']:,.0f} ms → "
                         f"GPU {top['gpu_ms']:,.2f} ms)")
        lines.append("")

    # ---- Performance figures section ----
    lines.append("## Performance figures\n")
    perf_figs = [
        ("fig01_speedup_grid.png",
         "Speedup per query, one panel per dataset."),
        ("fig02_speedup_vs_size.png",
         "Median and min/max speedup vs |E|. Each point is one dataset."),
        ("fig03_per_depth_time.png",
         "Stacked GPU time per depth. Most expansion work concentrates at "
         "the deepest levels."),
        ("fig04_frontier_growth.png",
         "Frontier size (log scale) vs depth, one line per query."),
        ("fig05_cpu_vs_gpu_scatter.png",
         "CPU ms vs GPU ms per query. Reference lines mark 100× and 1000×."),
        ("fig06_aggregate_speedup.png",
         "Aggregate speedup (Σ CPU / Σ GPU) per dataset."),
    ]
    for fname, caption in perf_figs:
        if (fig_dir / fname).exists():
            lines.append(f"### {fname}")
            lines.append(f"![{fname}](./{fname})")
            lines.append(f"_{caption}_\n")

    # ---- Per-dataset summary table ----
    if not perf_df.empty:
        lines.append("## Per-dataset summary\n")
        agg = (perf_df.groupby("dataset")
                       .agg(queries=("query", "count"),
                            cpu_total_ms=("cpu_ms", "sum"),
                            gpu_total_ms=("gpu_ms", "sum"),
                            median_speedup=("speedup", "median"),
                            max_speedup=("speedup", "max"))
                       .reset_index())
        agg["aggregate_speedup"] = agg["cpu_total_ms"] / agg["gpu_total_ms"]
        for col in ["cpu_total_ms", "gpu_total_ms"]:
            agg[col] = agg[col].round(2)
        for col in ["median_speedup", "max_speedup", "aggregate_speedup"]:
            agg[col] = agg[col].round(1)
        lines.append(agg.to_markdown(index=False, floatfmt=",.1f"))
        lines.append("")

    # ---- Profiling figures section ----
    if not wide.empty:
        lines.append("## GPU profiling (Nsight Compute)\n")
        prof_figs = [
            ("fig07_occupancy.png",
             "Distribution of achieved active-warp occupancy per launch."),
            ("fig08_memory_throughput.png",
             "DRAM, L1/TEX, and L2 throughput distributions."),
            ("fig09_compute_vs_memory.png",
             "Per-launch SM compute utilization vs memory utilization. "
             "Points below the diagonal indicate the kernel is memory-bound."),
            ("fig10_warp_states.png",
             "Top warp stall reasons by average cycles per issued instruction."),
            ("fig11_sm_utilization.png",
             "SM utilization (% of peak) across kernel launches."),
            ("fig12_roofline.png",
             "Arithmetic intensity vs throughput. Memory-bound kernels "
             "fall on the left of the knee."),
        ]
        for fname, caption in prof_figs:
            if (fig_dir / fname).exists():
                lines.append(f"### {fname}")
                lines.append(f"![{fname}](./{fname})")
                lines.append(f"_{caption}_\n")

        # Inline some headline kernel metrics.
        lines.append("### Key kernel metrics (mean across all profiled launches)\n")
        candidates = [
            ("Achieved active warps",
             ["sm__warps_active.avg.pct_of_peak_sustained_active",
              "Achieved Active Warps Per SM"]),
            ("Theoretical occupancy",
             ["Theoretical Occupancy", "Theoretical Active Warps Per SM"]),
            ("SM throughput (% of peak)",
             ["sm__throughput.avg.pct_of_peak_sustained_elapsed",
              "Compute (SM) Throughput"]),
            ("DRAM throughput (% of peak)",
             ["dram__throughput.avg.pct_of_peak_sustained_elapsed",
              "Memory Throughput"]),
            ("L1/TEX hit rate",
             ["l1tex__t_sector_hit_rate.pct", "L1/TEX Hit Rate"]),
            ("L2 hit rate",
             ["lts__t_sector_hit_rate.pct", "L2 Hit Rate"]),
            ("Average warps per scheduler",
             ["smsp__warps_active.avg.per_cycle_active",
              "Active Warps Per Scheduler"]),
            ("Block size",
             ["Block Size"]),
            ("Grid size",
             ["Grid Size"]),
        ]
        rows = []
        for label, patterns in candidates:
            col = find_metric_column(wide, *patterns)
            if col is None:
                continue
            v = pd.to_numeric(wide[col], errors="coerce").dropna()
            if v.empty:
                continue
            rows.append({
                "Metric": label,
                "ncu Column": col,
                "Mean": float(v.mean()),
                "Median": float(v.median()),
                "Min": float(v.min()),
                "Max": float(v.max()),
                "N": int(len(v)),
            })
        if rows:
            mdf = pd.DataFrame(rows)
            for c in ["Mean", "Median", "Min", "Max"]:
                mdf[c] = mdf[c].round(3)
            lines.append(mdf.to_markdown(index=False))
            lines.append("")

    # ---- Discussion ----
    lines.append("## Discussion\n")
    if not perf_df.empty:
        m = perf_df[perf_df["cpu_ms"] >= 5.0]
        if not m.empty:
            lines.append(
                f"Across {len(m)} non-trivial query runs (CPU ≥ 5 ms) the GPU "
                f"implementation achieves a median speedup of "
                f"**{m['speedup'].median():,.1f}×** and a maximum of "
                f"**{m['speedup'].max():,.1f}×**. The largest absolute "
                f"reductions in wall-clock time are on the deepest patterns "
                f"(`path_5`, `star_5`) where the CPU spends seconds-to-minutes "
                f"and the GPU completes in tens of milliseconds.\n"
            )

    if not wide.empty:
        sm_col = find_metric_column(wide,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "Compute (SM) Throughput")
        mem_col = find_metric_column(wide,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "Memory Throughput")
        if sm_col and mem_col:
            sm_mean = pd.to_numeric(wide[sm_col], errors="coerce").mean()
            mem_mean = pd.to_numeric(wide[mem_col], errors="coerce").mean()
            if pd.notna(sm_mean) and pd.notna(mem_mean):
                bound = "memory-bound" if mem_mean > sm_mean else "compute-bound"
                lines.append(
                    f"On average the kernel reaches **{sm_mean:.1f}%** of "
                    f"peak compute throughput and **{mem_mean:.1f}%** of "
                    f"peak DRAM throughput, indicating the workload is "
                    f"**{bound}**. This is consistent with a kernel whose "
                    f"inner loop is dominated by binary-search adjacency "
                    f"checks against the data CSR.\n"
                )

    lines.append(
        "**Plausible next optimizations** (in priority order):\n"
        "1. Warp-cooperative `is_edge_consistent_undir`: distribute the "
        "   prior-mappings loop across lanes; use `__ballot_sync` for early "
        "   exit on the first failed adjacency check.\n"
        "2. Pre-intersection of candidates with neighbors of already-mapped "
        "   pattern vertices, computed once per depth on the host. This "
        "   would shrink `cand_count` for all but the first depth.\n"
        "3. Streaming candidate tile through shared memory, reducing the "
        "   per-thread global memory traffic for the candidate flat array.\n"
        "4. Promise-based dynamic block allocation (project stretch goal).\n"
    )

    out.write_text("\n".join(lines))


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("profile_dir", nargs="?", default="profile_results",
                   help="Directory of ncu CSVs from profile_suite.sh")
    p.add_argument("--results-log", default=None,
                   help="Path to results.log from run_suite.sh "
                        "(default: <profile_dir>/../suite/results.log "
                        "or <profile_dir>/results.log)")
    p.add_argument("--out-dir", default="figures",
                   help="Output directory for figures and report")
    args = p.parse_args()

    pdir = Path(args.profile_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate results.log
    log_candidates = []
    if args.results_log:
        log_candidates.append(Path(args.results_log))
    log_candidates += [
        pdir / "results.log",
        pdir.parent / "suite" / "results.log",
        Path.cwd() / "suite" / "results.log",
        Path.cwd() / "results.log",
    ]
    results_log = next((c for c in log_candidates if c.exists()), None)

    print(f"profile dir : {pdir}")
    print(f"results log : {results_log}")
    print(f"out dir     : {out_dir}")

    # ---- Parse solver log ----
    datasets = {}
    if results_log:
        text = results_log.read_text(errors="replace")
        datasets = parse_results_log(text)
        print(f"  parsed {len(datasets)} datasets from results.log")
        for k, d in datasets.items():
            print(f"    {k}: {len(d['summary'])} queries, "
                  f"V={d['num_vertices']}, E={d['num_edges']}")
    else:
        print("  warning: no results.log found; performance figures will be skipped")

    # ---- Parse ncu CSVs ----
    ncu_dfs = []
    csv_files = sorted(pdir.glob("phase_*.csv"))
    print(f"\nfound {len(csv_files)} ncu CSVs:")
    for cf in csv_files:
        df = parse_ncu_csv(cf)
        if not df.empty:
            print(f"  {cf.name}: {len(df)} rows")
            ncu_dfs.append(df)
        else:
            print(f"  {cf.name}: EMPTY or unparseable")

    if ncu_dfs:
        all_long = pd.concat(ncu_dfs, ignore_index=True)
        # Bash script no longer filters by kernel name at the ncu level
        # (--kernel-name turned out to be flaky across versions). Filter here.
        if "Kernel Name" in all_long.columns:
            mask = all_long["Kernel Name"].astype(str).str.contains(
                "expand_frontier_kernel", na=False)
            n_total = len(all_long)
            all_long = all_long[mask].copy()
            print(f"  filtered to expand_frontier_kernel: "
                  f"{len(all_long)}/{n_total} rows kept")
        wide = pivot_metrics_wide(all_long)
        print(f"  wide form: {len(wide)} kernel launches, "
              f"{len(wide.columns)} metric columns")
    else:
        wide = pd.DataFrame()

    # ---- Generate performance figures ----
    if datasets:
        print("\nperformance figures:")
        for name, fn in [
            ("fig01_speedup_grid.png", fig01_speedup_grid),
            ("fig02_speedup_vs_size.png", fig02_speedup_vs_size),
            ("fig03_per_depth_time.png", fig03_per_depth_time),
            ("fig04_frontier_growth.png", fig04_frontier_growth),
            ("fig05_cpu_vs_gpu_scatter.png", fig05_cpu_vs_gpu_scatter),
            ("fig06_aggregate_speedup.png", fig06_aggregate_speedup),
        ]:
            fp = out_dir / name
            try:
                fn(datasets, fp)
            except Exception as e:
                print(f"  failed {name}: {e}")
                continue
            if fp.exists():
                print(f"  wrote {name}")
            else:
                print(f"  skipped {name} (insufficient data)")

    # ---- Generate profiling figures ----
    if not wide.empty:
        print("\nprofiling figures:")
        for name, fn in [
            ("fig07_occupancy.png", fig07_occupancy),
            ("fig08_memory_throughput.png", fig08_memory_throughput),
            ("fig09_compute_vs_memory.png", fig09_compute_vs_memory),
            ("fig10_warp_states.png", fig10_warp_states),
            ("fig11_sm_utilization.png", fig11_sm_utilization),
            ("fig12_roofline.png", fig12_roofline),
        ]:
            fp = out_dir / name
            try:
                fn(wide, fp)
            except Exception as e:
                print(f"  failed {name}: {e}")
                continue
            if fp.exists():
                print(f"  wrote {name}")
            else:
                print(f"  skipped {name} (metric not in profile)")

    # ---- Tables ----
    if datasets:
        rows = []
        for k, d in datasets.items():
            df = d["summary"]
            if df.empty:
                continue
            for _, r in df.iterrows():
                rows.append({
                    "dataset": k,
                    "num_vertices": d["num_vertices"],
                    "num_edges": d["num_edges"],
                    **r.to_dict(),
                })
        if rows:
            perf = pd.DataFrame(rows)
            perf.to_csv(out_dir / "summary_perf.csv", index=False)
            print(f"  wrote summary_perf.csv ({len(perf)} rows)")

    if not wide.empty:
        wide.to_csv(out_dir / "summary_kernels.csv", index=False)
        print(f"  wrote summary_kernels.csv ({len(wide)} launches)")

    # ---- Report ----
    write_report(datasets, wide, out_dir / "report.md", out_dir)
    print(f"\nwrote {out_dir / 'report.md'}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
