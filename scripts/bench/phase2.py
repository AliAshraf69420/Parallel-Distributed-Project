#!/usr/bin/env python3
"""Phase 2 benchmark runner (MPI). Same method as Phase 1: one discarded warm-up run, then
--runs measured runs per configuration (median reported), --preload on every run, and the
first measured run of every configuration checked bit-exactly against the reference.

Two environments
  host     mpirun on this machine: one node, ranks pinned to cores, shared-memory transport
  cluster  the 4-node Docker cluster (scripts/mpi/cluster.sh up): 2 cores per node, shared
           memory inside a node, TCP between nodes

Experiments (select with --only; default: all that the environment allows)
  strong     P = 1, 2, 4, 8 ranks on 100MB / 500MB / full        (host and cluster)
  weak       P ranks on the P x 700 MiB set, and 1 rank on each   (host and cluster)
  partition  P = 8 on full with --partition work | count | pixels (host): per-rank balance
  overlap    --collect send vs isend on 500MB with 1, 2, 3 and 7 workers (host and cluster)
  hybrid     ranks x OpenMP threads = 8 on full: 1x8, 2x4, 4x2, 8x1 (host)
  pingpong   round-trip time vs message size: host, same container, across containers

Output: results/timings/phase2/*.csv (raw rows), ranks_*.csv (per-rank breakdowns),
        pingpong_*.csv, env.json

Usage (from the repo root, conda env active, after `make`):
  python scripts/bench/phase2.py [--only ...] [--runs 3] [--no-cluster]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "timings" / "phase2"
SCRATCH = Path("results/runs/bench2")          # relative: also valid inside the containers
CLUSTER = str(REPO / "scripts" / "mpi" / "cluster.sh")

SETS = ["100MB", "500MB", "full"]
PROCS = [1, 2, 4, 8]
WEAK_BASE = 700
EXPERIMENTS = ["strong", "weak", "partition", "overlap", "hybrid", "pingpong"]


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, **kw)


def launcher(env: str, p: int, pe: int = 1) -> list[str]:
    """Command prefix that starts p ranks in the given environment."""
    if env == "host":
        cmd = ["mpirun", "-n", str(p), "--bind-to", "core"]
        if pe > 1:
            cmd += ["--map-by", f"slot:PE={pe}", "-x", "OMP_PROC_BIND=close", "-x", "OMP_PLACES=cores"]
        return cmd + ["build/mpi"]
    # cluster: ranks are spread round-robin over the nodes, so P = 2 already crosses the
    # network (P = 2: 2 nodes, P = 4: 4 nodes x 1 rank, P = 8: 4 nodes x 2 ranks)
    return [CLUSTER, "spread", "-n", str(p), "build-cluster/mpi"]


def config(env: str, p: int, set_name: str, csv: Path, runs: int, extra: list[str], label: str,
           pe: int = 1, keep_ranks: bool = False) -> None:
    out = SCRATCH / label.replace(" ", "_").replace(",", "-")
    base = launcher(env, p, pe) + ["--set", set_name, "--preload", "--out", str(out), "-q", *extra]
    t0 = time.perf_counter()
    for k in range(-1, runs):                                   # k = -1 is the warm-up
        cmd = base if k < 0 else base + ["--csv", str(csv.relative_to(REPO)), "--run", str(k)]
        res = sh(cmd)
        if res.returncode != 0:
            sys.exit(f"command failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}{res.stderr}")
        if k == 0:
            chk = sh([sys.executable, "tests/check.py", "--set", set_name, "--run", str(out)])
            if chk.returncode != 0:
                sys.exit(f"correctness check failed for {' '.join(cmd)}:\n{chk.stdout}{chk.stderr}")
            if keep_ranks:
                shutil.copy(REPO / out / "ranks.csv", OUT / f"ranks_{label.replace(' ', '_')}.csv")
    print(f"  {label:<44} {time.perf_counter() - t0:6.1f} s", flush=True)


def fresh(name: str) -> Path:
    path = OUT / name
    if path.exists():
        path.unlink()
    return path


def cluster_ready() -> bool:
    r = sh([CLUSTER, "run", "-n", "8", "cat", "/etc/hostname"])
    return r.returncode == 0 and len(set(r.stdout.split())) == 4


def environment(cluster: bool) -> dict:
    def out(cmd: str) -> str:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "cpu": out("lscpu | sed -n 's/^Model name: *//p'"),
        "cores": out("lscpu | sed -n 's/^Core(s) per socket: *//p'"),
        "kernel": platform.release(),
        "mpi": out("mpirun --version | head -1"),
        "compiler": out("g++ --version | head -1"),
        "cxxflags": "-O3 -march=native -std=c++20 -ffp-contract=off",
        "host_launch": "mpirun --bind-to core (shared-memory transport)",
        "cluster": ("4 Docker containers x 2 physical cores (cpuset), bridge network, TCP between "
                    "nodes, shared memory inside a node, MPI binding off") if cluster else "not used",
        "cluster_mpi": out(f"{CLUSTER} run -n 1 mpirun --version | head -1") if cluster else "",
        "load_average_at_start": os.getloadavg(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-cluster", action="store_true", help="host experiments only")
    args = ap.parse_args()

    if not (REPO / "build" / "mpi").exists():
        sys.exit("build/mpi missing: install OpenMPI and run `make`")
    cluster = not args.no_cluster and cluster_ready()
    if not args.no_cluster and not cluster:
        print("warning: Docker cluster not running (scripts/mpi/cluster.sh up); "
              "host experiments only", file=sys.stderr)
    envs = ["host"] + (["cluster"] if cluster else [])
    weak_sets = [f"{WEAK_BASE * p}MB" for p in PROCS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env.json").write_text(json.dumps(environment(cluster), indent=2))
    load = os.getloadavg()[0]
    if load > 1.5:
        print(f"warning: load average is {load:.1f}; close other programs for clean timings",
              file=sys.stderr)
    t_all = time.perf_counter()

    if "strong" in args.only:
        for env in envs:
            print(f"strong scaling ({env})")
            csv = fresh(f"strong_{env}.csv")
            for s in SETS:
                for p in PROCS:
                    config(env, p, s, csv, args.runs, [], f"{env} {s} P={p}")

    if "weak" in args.only:
        for env in envs:
            print(f"weak scaling ({env})")
            csv = fresh(f"weak_{env}.csv")
            for p, s in zip(PROCS, weak_sets):
                config(env, 1, s, csv, args.runs, [], f"{env} {s} P=1")
                if p > 1:
                    config(env, p, s, csv, args.runs, [], f"{env} {s} P={p}")

    if "partition" in args.only:
        print("partition (host, P=8, full)")
        csv = fresh("partition_host.csv")
        for part in ("work", "count", "pixels"):
            config("host", 8, "full", csv, args.runs, ["--partition", part],
                   f"partition {part}", keep_ranks=True)

    if "overlap" in args.only:
        for env in envs:
            print(f"communication/computation overlap ({env})")
            csv = fresh(f"overlap_{env}.csv")
            for p in (2, 3, 4, 8):                      # 1, 2, 3, 7 workers + the collector
                for mode in ("send", "isend"):
                    config(env, p, "500MB", csv, args.runs, ["--collect", mode],
                           f"overlap {env} P={p} {mode}", keep_ranks=True)

    if "hybrid" in args.only:
        print("hybrid MPI x OpenMP (host, full)")
        csv = fresh("hybrid_host.csv")
        for p, t in ((1, 8), (2, 4), (4, 2), (8, 1)):
            config("host", p, "full", csv, args.runs, ["-t", str(t)], f"hybrid {p}x{t}", pe=t)

    if "pingpong" in args.only:
        print("ping-pong")
        jobs = [("host", ["mpirun", "-n", "2", "--bind-to", "core", "build/pingpong", "--label", "host"])]
        if cluster:
            jobs += [("same_node", [CLUSTER, "run", "-n", "2", "build-cluster/pingpong", "--label", "same-node"]),
                     ("cross_node", [CLUSTER, "spread", "-n", "2", "build-cluster/pingpong", "--label", "cross-node"])]
        for name, cmd in jobs:
            res = sh(cmd)
            if res.returncode != 0:
                sys.exit(f"pingpong failed: {' '.join(cmd)}\n{res.stderr}")
            (OUT / f"pingpong_{name}.csv").write_text(res.stdout)
            print(f"  pingpong {name}")

    print(f"done in {time.perf_counter() - t_all:.0f} s; raw CSVs in {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
