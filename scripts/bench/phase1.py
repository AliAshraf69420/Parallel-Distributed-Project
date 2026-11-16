#!/usr/bin/env python3
"""Phase 1 benchmark runner: every timing experiment of the report, in one command.

Experiments (select with --only; default: all)
  strong    seq, and omp + threads (C++ pipeline) with 1/2/4/8 threads (+16 = SMT),
            on 100MB / 500MB / full
  schedule  omp, 4 and 8 threads, 100MB and full, orders id / area-desc,
            schedules static, static,1, dynamic,1, dynamic,16, guided
  weak      Gustafson: omp with p threads on the p x 700 MiB set (p = 1, 2, 4, 8),
            plus 1 thread on every one of those sets (for scaled speedup)
  records   seq on full with per-image timing (how uneven is the work per image?)

Method (same in every phase, see docs/PLAN.md section 5.4)
  * every configuration: 1 discarded warm-up run, then --runs measured runs (default 3);
    report the median
  * --preload on every run (shards are faulted into the page cache before timing)
  * OMP_PROC_BIND=close, OMP_PLACES=cores: threads 1..8 each get their own physical core
  * the first measured run of every configuration is checked bit-exactly against the
    reference (tests/check.py); any FAIL aborts the benchmark

Output: results/timings/phase1_<experiment>.csv (raw rows, one per run),
        results/timings/phase1_records_full.csv, results/timings/phase1_env.json

Usage (from the repo root, conda env active, after `make`):
  python scripts/bench/phase1.py [--only strong schedule weak records] [--runs 3]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TIMINGS = REPO / "results" / "timings"
SCRATCH = REPO / "results" / "runs" / "bench"
ENV = {**os.environ, "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}

STRONG_SETS = ["100MB", "500MB", "full"]
STRONG_THREADS = [1, 2, 4, 8, 16]
SCHED_SETS = ["100MB", "full"]
SCHED_THREADS = [4, 8]
SCHEDULES = ["static", "static,1", "dynamic,1", "dynamic,16", "guided"]
ORDERS = ["id", "area-desc"]
WEAK_BASE_MIB = 700
WEAK_P = [1, 2, 4, 8]


def run(cmd: list[str], check_set: str | None = None, out: Path | None = None) -> None:
    res = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr}")
    if check_set:
        chk = subprocess.run([sys.executable, "tests/check.py", "--set", check_set, "--run", str(out)],
                             cwd=REPO, capture_output=True, text=True)
        if chk.returncode != 0:
            sys.exit(f"correctness check failed for {' '.join(cmd)}:\n{chk.stdout}{chk.stderr}")


def config(binary: str, set_name: str, csv: Path, runs: int, extra: list[str], label: str) -> None:
    out = SCRATCH / label.replace(" ", "_").replace(",", "-")
    base = [f"build/{binary}", "--set", set_name, "--preload", "--out", str(out), "-q", *extra]
    t0 = time.perf_counter()
    run(base)                                                     # warm-up, not recorded
    for k in range(runs):
        run(base + ["--csv", str(csv), "--run", str(k)],
            check_set=set_name if k == 0 else None, out=out)
    print(f"  {label:<48} {time.perf_counter() - t0:6.1f} s", flush=True)


def fresh(path: Path) -> Path:
    if path.exists():
        path.unlink()
    return path


def environment() -> dict:
    def sh(cmd: str) -> str:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return r.stdout.strip()
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "cpu": sh("lscpu | sed -n 's/^Model name: *//p'"),
        "cores": sh("lscpu | sed -n 's/^Core(s) per socket: *//p'"),
        "threads_per_core": sh("lscpu | sed -n 's/^Thread(s) per core: *//p'"),
        "caches": sh("lscpu | grep -E '^(L1d|L2|L3)' | tr -s ' '").splitlines(),
        "ram": sh("free -h | awk '/^Mem/ {print $2}'"),
        "governor": gov.read_text().strip() if gov.exists() else "unknown",
        "kernel": platform.release(),
        "compiler": sh("g++ --version | head -1"),
        "cxxflags": "-O3 -march=native -std=c++20 -ffp-contract=off",
        "omp_env": {k: ENV[k] for k in ("OMP_PROC_BIND", "OMP_PLACES")},
        "load_average_at_start": os.getloadavg(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=["strong", "schedule", "weak", "records"],
                    default=["strong", "schedule", "weak", "records"])
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    for b in ("seq", "omp", "threads"):
        if not (REPO / "build" / b).exists():
            sys.exit(f"build/{b} missing: run `make` first")
    weak_sets = [f"{WEAK_BASE_MIB * p}MB" for p in WEAK_P]
    needed = set(STRONG_SETS) | set(SCHED_SETS) | (set(weak_sets) if "weak" in args.only else set())
    for s in sorted(needed):
        if not (REPO / "results" / "reference" / s / "stats.json").exists():
            sys.exit(f"missing reference for set {s}: run make_subsets.py / reference.py (README)")

    TIMINGS.mkdir(parents=True, exist_ok=True)
    load = os.getloadavg()[0]
    if load > 1.5:
        print(f"warning: load average is {load:.1f}; close other programs for clean timings",
              file=sys.stderr)
    (TIMINGS / "phase1_env.json").write_text(json.dumps(environment(), indent=2))
    t_all = time.perf_counter()

    if "strong" in args.only:
        print("strong scaling")
        csv = fresh(TIMINGS / "phase1_strong.csv")
        for s in STRONG_SETS:
            config("seq", s, csv, args.runs, [], f"seq {s}")
            for t in STRONG_THREADS:
                config("omp", s, csv, args.runs, ["-t", str(t), "--schedule", "static"],
                       f"omp {s} t={t}")
            for t in STRONG_THREADS:
                config("threads", s, csv, args.runs, ["-t", str(t)], f"threads {s} t={t}")

    if "schedule" in args.only:
        print("scheduling")
        csv = fresh(TIMINGS / "phase1_schedule.csv")
        for s in SCHED_SETS:
            for order in ORDERS:
                for t in SCHED_THREADS:
                    for sch in SCHEDULES:
                        config("omp", s, csv, args.runs,
                               ["-t", str(t), "--schedule", sch, "--order", order],
                               f"omp {s} {order} t={t} {sch}")

    if "weak" in args.only:
        print("weak scaling (Gustafson)")
        csv = fresh(TIMINGS / "phase1_weak.csv")
        for p, s in zip(WEAK_P, weak_sets):
            config("omp", s, csv, args.runs, ["-t", "1", "--schedule", "static"], f"omp {s} t=1")
            if p > 1:
                config("omp", s, csv, args.runs, ["-t", str(p), "--schedule", "static"],
                       f"omp {s} t={p}")

    if "records" in args.only:
        print("per-image work distribution")
        rt = TIMINGS / "phase1_records_full.csv"
        out = SCRATCH / "records"
        run(["build/seq", "--set", "full", "--preload", "--out", str(out), "-q"])  # warm-up
        run(["build/seq", "--set", "full", "--preload", "--out", str(out), "-q",
             "--record-times", str(rt)], check_set="full", out=out)
        print(f"  seq full per-image times -> {rt.relative_to(REPO)}")

    print(f"done in {time.perf_counter() - t_all:.0f} s; raw CSVs in {TIMINGS.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
