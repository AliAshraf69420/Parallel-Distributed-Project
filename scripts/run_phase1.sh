#!/usr/bin/env bash
# Phase 1, end to end: build, correctness, benchmarks, memory experiments, plots, report.
#
# Usage (from anywhere; the conda env `sw401` must be active, preprocessing done, see README 1-4):
#   scripts/run_phase1.sh                  every step below, in order (~15 min)
#   scripts/run_phase1.sh plots report     only the named steps
#   scripts/run_phase1.sh --runs 5 bench   pass --runs to the benchmark runner (default 3)
#
# Steps
#   build    make                         -> build/{seq,omp,threads,memexp}
#   test     make test                    seq, omp, threads on 100MB, checked bit-exactly
#   bench    scripts/bench/phase1.py      strong, schedule, weak, per-image times
#   memexp   build/memexp                 wss, stride, resize-order, false-sharing
#   perf     perf stat                    cache counters (resize order, false sharing) and
#                                         cycles/instructions of omp -t 1 vs -t 8 on full
#   plots    plots/phase1.py              -> plots/phase1/*.png, results/timings/phase1_summary.md
#   report   make report                  -> report.pdf
#
# bench, memexp and perf overwrite the committed results in results/timings/. The report
# quotes those numbers, so rerun plots and report afterwards (the default order does) and
# reread the report if the numbers moved. Close other programs first: timings are sensitive
# to background load. perf is skipped with a warning if it is missing or not permitted.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ALL_STEPS=(build test bench memexp perf plots report)
RUNS=3
STEPS=()
while (($#)); do
    case "$1" in
        --runs) RUNS="${2:?--runs needs a value}"; shift 2 ;;
        -h|--help) sed -n '2,23p' "$REPO/scripts/run_phase1.sh" | sed 's/^# \{0,1\}//'; exit 0 ;;
        build|test|bench|memexp|perf|plots|report) STEPS+=("$1"); shift ;;
        *) echo "unknown step or option: $1 (try --help)" >&2; exit 2 ;;
    esac
done
((${#STEPS[@]})) || STEPS=("${ALL_STEPS[@]}")

MEM="${MEMEXP_DIR:-results/timings/memexp}"   # override to keep the committed results
export OMP_PROC_BIND=close OMP_PLACES=cores

step() { printf '\n==== %s ====\n' "$*"; }
want() { local s; for s in "${STEPS[@]}"; do [[ $s == "$1" ]] && return 0; done; return 1; }
needs_bin() { [[ -x build/$1 ]] || { echo "build/$1 missing: run the build step first" >&2; exit 1; }; }

# ---- prerequisites ----------------------------------------------------------------------
if ! python -c 'import numpy, matplotlib' 2>/dev/null; then
    echo "python cannot import numpy/matplotlib: run 'conda activate sw401' first" >&2
    exit 1
fi
if want test || want bench || want memexp || want perf; then
    for s in 100MB 500MB full; do
        [[ -f data/processed/$s/index.bin ]] || {
            echo "data/processed/$s missing: run the preprocessing steps (README 1-4, or 'make data')" >&2
            exit 1; }
    done
fi
if want bench; then
    for s in 700MB 1400MB 2800MB 5600MB; do
        [[ -f data/processed/$s/index.bin ]] || {
            echo "weak-scaling set data/processed/$s missing: run 'make data'" >&2; exit 1; }
    done
    load=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$load" 'BEGIN { exit !(l > 2) }'; then
        echo "warning: load average is $load; close other programs for stable timings" >&2
    fi
fi

START=$SECONDS

# ---- steps ----------------------------------------------------------------------------
if want build; then
    step "build"
    make -j"$(nproc)"
fi

if want test; then
    step "test: bit-exact check on 100MB"
    make test
fi

if want bench; then
    step "bench: strong, schedule, weak, per-image (runs=$RUNS)"
    python scripts/bench/phase1.py --runs "$RUNS"
fi

if want memexp; then
    step "memexp: memory hierarchy and false sharing"
    needs_bin memexp
    mkdir -p "$MEM"
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    # Write to a temp dir first so a failed run leaves the previous results intact.
    build/memexp wss                          > "$tmp/wss.csv"
    build/memexp stride                       > "$tmp/stride.csv"
    build/memexp resize-order --set 500MB     > "$tmp/resize_order.csv"
    build/memexp false-sharing --set 500MB -t 8 > "$tmp/false_sharing.csv"
    for f in "$tmp"/*.csv; do
        mv "$f" "$MEM/"
        echo "-- $MEM/$(basename "$f")"
        column -s, -t < "$MEM/$(basename "$f")"
    done
fi

if want perf; then
    step "perf: hardware counters"
    needs_bin memexp
    needs_bin omp
    if ! command -v perf >/dev/null; then
        echo "warning: perf not installed, skipping (the report's perf tables keep their old values)" >&2
    elif ! perf stat -e cycles true >/dev/null 2>&1; then
        echo "warning: perf cannot read counters (try: sudo sysctl kernel.perf_event_paranoid=1), skipping" >&2
    else
        mkdir -p "$MEM"
        EV=cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses
        for v in row col; do
            perf stat -x, -e "$EV" -o "$MEM/perf_resize_$v.csv" \
                build/memexp resize-order --set 500MB --variant "$v" --reps 1 >/dev/null
        done
        for v in packed padded local; do
            perf stat -x, -e "$EV" -o "$MEM/perf_fs_$v.csv" \
                build/memexp false-sharing --set 500MB -t 8 --variant "$v" --reps 1 >/dev/null
        done
        # Same instruction count at 1 and 8 threads; the cycle ratio shows the clock drop.
        out=results/runs/bench/freq
        build/omp --set full -t 1 --preload -q --out "$out" >/dev/null   # warm-up
        for t in 1 8; do
            perf stat -x, -e cycles,instructions,task-clock -o "$MEM/perf_omp_full_t$t.csv" \
                build/omp --set full -t "$t" --schedule static --preload -q --out "$out"
        done
        for f in "$MEM"/perf_*.csv; do
            printf '%-28s' "$(basename "$f")"
            awk -F, '!/^#/ && NF > 2 { printf "  %s=%s", $3, $1 } END { print "" }' "$f"
        done
    fi
fi

if want plots; then
    step "plots and summary tables"
    python plots/phase1.py
fi

if want report; then
    step "report"
    if command -v latexmk >/dev/null; then
        make report
    else
        echo "warning: latexmk not installed, skipping the report" >&2
    fi
fi

printf '\nPhase 1 steps done (%s) in %d s.\n' "${STEPS[*]}" $((SECONDS - START))
