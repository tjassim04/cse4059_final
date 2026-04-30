#!/usr/bin/env bash
# profile_suite.sh
#
# Comprehensive Nsight Compute (ncu) profiling driver for the GPU subgraph
# matching solver. Captures both pure-GPU kernel performance and how that
# performance scales across graph sizes.
#
# Phases:
#   A: All 15 queries on 02_er_n1k_d6   (--set detailed, fast medium graph)
#   B: All 15 queries on 03_er_n2k_d8   (--set detailed, primary target)
#   C: Subset queries across 3 sizes    (--set basic, scaling comparison)
#   D: Hot kernel source-level dive     (--set full, first 20 launches)
#   E: Roofline analysis                (SpeedOfLight + roofline sections)
#
# Phases B and D are the most expensive and produce the bulk of useful data.
# A and C are cheap context. E is needed for the roofline figure.
#
# Each phase produces:
#   <out_dir>/<phase>.csv          - ncu CSV report (long-form metrics)
#   <out_dir>/<phase>.solver.log   - solver stdout/stderr captured during run
#
# Plus, on first invocation:
#   <out_dir>/ncu_version.txt
#   <out_dir>/profile_manifest.txt
#
# Usage:
#   ./profile_suite.sh [solver_binary] [suite_dir] [out_dir]
#
# Defaults:
#   solver_binary = ./Subgraph_Solution
#   suite_dir     = ./suite
#   out_dir       = ./profile_results
#
# Expected runtime: ~20-30 minutes on an A100, depending on dataset sizes.

set -uo pipefail

SOLVER_BIN="${1:-./Subgraph_Solution}"
SUITE_DIR="${2:-./suite}"
OUT_DIR="${3:-./profile_results}"

# === Validate ===
if [ ! -x "$SOLVER_BIN" ]; then
    echo "ERROR: solver binary not found or not executable: $SOLVER_BIN" >&2
    exit 1
fi
if [ ! -d "$SUITE_DIR" ]; then
    echo "ERROR: suite directory not found: $SUITE_DIR" >&2
    exit 1
fi

# === Locate ncu ===
NCU="$(command -v ncu 2>/dev/null || true)"
if [ -z "$NCU" ]; then
    for candidate in \
        /usr/local/cuda/bin/ncu \
        /opt/nvidia/nsight-compute/*/ncu \
        /usr/local/NVIDIA-Nsight-Compute-*/ncu; do
        if [ -x "$candidate" ]; then
            NCU="$candidate"
            break
        fi
    done
fi
if [ -z "$NCU" ]; then
    echo "ERROR: nsight compute (ncu) not found." >&2
    echo "       Try 'module load nsight-compute' or set PATH." >&2
    exit 1
fi

# === Resolve absolute paths (we will cd into per-dataset dirs) ===
mkdir -p "$OUT_DIR"
SOLVER_ABS="$(cd "$(dirname "$SOLVER_BIN")" && pwd)/$(basename "$SOLVER_BIN")"
SUITE_ABS="$(cd "$SUITE_DIR" && pwd)"
OUT_ABS="$(cd "$OUT_DIR" && pwd)"

echo "==================================================================="
echo "  profile_suite.sh"
echo "==================================================================="
echo "  ncu       : $NCU"
echo "  solver    : $SOLVER_ABS"
echo "  suite     : $SUITE_ABS"
echo "  out       : $OUT_ABS"
echo "  started   : $(date)"
echo "==================================================================="

# Capture environment metadata for the report.
{
    echo "ncu_path=$NCU"
    "$NCU" --version 2>&1 || true
    echo
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || true
} > "$OUT_ABS/ncu_version.txt"

# === Sanity check: make sure ncu accepts our minimal flag combo ===
# Catches version-incompatibility issues (renamed flags, etc.) immediately
# instead of letting every phase silently fail.
echo
echo "Sanity check: ncu --csv --log-file ... --set basic <prog> ..."
SANITY_OUT="$OUT_ABS/_sanity_check.csv"
SANITY_LOG="$OUT_ABS/_sanity_check.log"
rm -f "$SANITY_OUT" "$SANITY_LOG"
# Pick the smallest dataset for the sanity check.
SANITY_DS="$(ls "$SUITE_ABS" | grep -E '^[0-9]+_' | sort | head -1)"
if [ -n "$SANITY_DS" ] && [ -d "$SUITE_ABS/$SANITY_DS" ]; then
    if (cd "$SUITE_ABS/$SANITY_DS" && \
        "$NCU" --csv --log-file "$SANITY_OUT" --set basic --launch-count 1 \
               "$SOLVER_ABS" data_edges.txt metadata.txt queries.txt \
               > "$SANITY_LOG" 2>&1); then
        echo "  ok: ncu accepts our flags"
    else
        echo "  FAIL: ncu rejected the minimal command. First 25 lines of log:"
        echo "  ---"
        head -25 "$SANITY_LOG" | sed 's/^/    /'
        echo "  ---"
        echo "  Aborting. Inspect $SANITY_LOG for full details."
        exit 1
    fi
else
    echo "  warning: no dataset found for sanity check; proceeding anyway"
fi

START_TIME=$SECONDS

# ===================================================================
# Helpers
# ===================================================================

# Extract a few queries by name from a master queries.txt.
make_subset() {
    local src="$1"; shift
    local dst="$1"; shift
    : > "$dst"
    for q in "$@"; do
        if ! grep "^$q " "$src" >> "$dst" 2>/dev/null; then
            echo "    warn: $q not found in $src"
        fi
    done
}

# Run one profiling phase. Captures CSV report + solver log.
# Args:  <label> <dataset_subdir> <queries_filename> [extra ncu args...]
run_phase() {
    local label="$1"; shift
    local dataset="$1"; shift
    local queries="$1"; shift

    local dataset_dir="$SUITE_ABS/$dataset"
    if [ ! -d "$dataset_dir" ]; then
        echo "[$label] SKIP: $dataset not present"
        return 0
    fi
    if [ ! -f "$dataset_dir/$queries" ]; then
        echo "[$label] SKIP: $queries not found in $dataset"
        return 0
    fi

    local csv_path="$OUT_ABS/${label}.csv"
    local log_path="$OUT_ABS/${label}.solver.log"

    echo
    echo "[$label] dataset=$dataset queries=$queries"
    echo "         extra opts: $*"
    local phase_start=$SECONDS

    pushd "$dataset_dir" > /dev/null

    # CSV report -> --log-file. Solver and ncu chatter -> redirected stdio.
    # We swallow ncu's own non-zero exit (sometimes it complains about kernels
    # that didn't run) so one bad phase doesn't kill the whole script.
    #
    # IMPORTANT: ncu's argument parser does NOT recognize "--" as the
    # end-of-options separator (unlike GNU getopt). Including "--" causes
    # ncu to treat the empty option name as ambiguous against every flag.
    # The program is identified positionally as the first non-flag argument.
    "$NCU" --csv \
           --log-file "$csv_path" \
           "$@" \
           "$SOLVER_ABS" data_edges.txt metadata.txt "$queries" \
           > "$log_path" 2>&1 \
        && echo "  status: ok" \
        || echo "  status: ncu returned non-zero (continuing)"

    popd > /dev/null

    local elapsed=$((SECONDS - phase_start))
    echo "  elapsed: ${elapsed}s   csv: $(basename "$csv_path")"
}

# ===================================================================
# Phase A: All queries on 02_er_n1k_d6, --set detailed
# Fast warmup. Establishes that the pipeline works and gives full metrics
# for small-but-non-trivial workloads.
# ===================================================================
echo
echo "##### Phase A: 02_er_n1k_d6, all queries, --set detailed #####"
run_phase "phase_a_02_er_n1k_d6_detailed" \
    "02_er_n1k_d6" "queries.txt" \
    --set detailed

# ===================================================================
# Phase B: All queries on 03_er_n2k_d8, --set detailed
# Primary profiling target. Largest dataset where path_5 etc. complete in
# reasonable time under ncu overhead.
# ===================================================================
echo
echo "##### Phase B: 03_er_n2k_d8, all queries, --set detailed #####"
run_phase "phase_b_03_er_n2k_d8_detailed" \
    "03_er_n2k_d8" "queries.txt" \
    --set detailed

# ===================================================================
# Phase C: Subset of queries on three different graph sizes
# Lets us correlate kernel metrics with input size. --set basic keeps it
# fast since we just need launch + occupancy + speed-of-light counters.
# ===================================================================
echo
echo "##### Phase C: cross-size scaling comparison, --set basic #####"
SUBSET_QUERIES=("triangle_3" "path_3" "path_4" "cycle_4" "star_4")

for ds in 01_motif_n300 02_er_n1k_d6 03_er_n2k_d8; do
    if [ -d "$SUITE_ABS/$ds" ]; then
        subset_path="$SUITE_ABS/$ds/_profile_subset.txt"
        make_subset "$SUITE_ABS/$ds/queries.txt" "$subset_path" \
            "${SUBSET_QUERIES[@]}"
        run_phase "phase_c_${ds}_basic" "$ds" "_profile_subset.txt" \
            --set basic
    fi
done

# ===================================================================
# Phase D: Source-level deep dive on the hot kernel
# We restrict to path_5 on 03_er_n2k_d8 and capture the first 20 kernel
# launches with --set full. This includes SourceCounters which gives us
# per-line-of-code stalls / inefficiency markers.
# ===================================================================
echo
echo "##### Phase D: source-level deep dive, --set full (first 20 launches) #####"
make_subset "$SUITE_ABS/03_er_n2k_d8/queries.txt" \
    "$SUITE_ABS/03_er_n2k_d8/_profile_path5.txt" "path_5"
run_phase "phase_d_03_er_n2k_d8_path5_full" \
    "03_er_n2k_d8" "_profile_path5.txt" \
    --set full --launch-count 20 \
    --import-source on

# ===================================================================
# Phase E: Roofline analysis on the subset queries
# SpeedOfLight_RooflineChart needs both compute and memory metrics. Run on
# a focused subset so we get clean per-kernel points.
# ===================================================================
echo
echo "##### Phase E: roofline analysis #####"
run_phase "phase_e_03_er_n2k_d8_roofline" \
    "03_er_n2k_d8" "_profile_subset.txt" \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section LaunchStats \
    --section Occupancy \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis

# ===================================================================
# Manifest + summary
# ===================================================================
TOTAL_ELAPSED=$((SECONDS - START_TIME))
{
    echo "profile_suite.sh manifest"
    echo "generated: $(date)"
    echo "elapsed:   ${TOTAL_ELAPSED}s"
    echo "solver:    $SOLVER_ABS"
    echo "suite:     $SUITE_ABS"
    echo
    echo "Files:"
    ls -la "$OUT_ABS"/*.csv 2>/dev/null
} > "$OUT_ABS/profile_manifest.txt"

echo
echo "==================================================================="
echo "  profile_suite.sh complete"
echo "  elapsed: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s)"
echo "  output : $OUT_ABS"
echo "==================================================================="
echo
echo "Generated CSVs:"
ls -la "$OUT_ABS"/*.csv 2>/dev/null | awk '{print "  " $NF " (" $5 " bytes)"}' || true
echo
echo "Next step:"
echo "    python3 analyze_results.py $OUT_ABS \\"
echo "        --results-log $SUITE_ABS/results.log \\"
echo "        --out-dir figures"
