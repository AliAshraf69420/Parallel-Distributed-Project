#!/usr/bin/env python3
"""Phase 2 plots and tables from results/timings/phase2/ (run scripts/bench/phase2.py first).

Writes to plots/phase2/:
  strong.png     speedup vs processes on the full set: MPI on the host, MPI on the Docker
                 cluster, and the Phase 1 OpenMP result
  weak.png       scaled speedup on the p x 700 MiB sets (host and cluster)
  pingpong.png   round-trip time vs message size (host, same node, across nodes)
  overlap.png    time each worker spends in send/wait: MPI_Send vs MPI_Isend
and results/timings/phase2/summary.md (all tables for the report, medians of the runs).

Usage: python plots/phase2.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
T2 = REPO / "results" / "timings" / "phase2"
T1 = REPO / "results" / "timings"
OUT = REPO / "plots" / "phase2"

# Reference palette (light mode): first three categorical slots, ink for all text.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
MARK = dict(marker="o", markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compute(r: dict) -> float:
    return float(r["t_pass1"]) + float(r["t_merge"]) + float(r["t_pass2"])


def med(rows: list[dict], key, value) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(value(r))
    return {k: statistics.median(v) for k, v in groups.items()}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  plots/phase2/{name}")


def end_labels(ax, ends: list[list], x) -> None:
    """Direct labels at the line ends, nudged apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = 0.07 * (hi - lo)
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for y, text in ends:
        ax.annotate(text, (x, y), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK2, annotation_clip=False)


def serial_baseline() -> dict:
    """Sequential compute/total time per set, from the Phase 1 run."""
    rows = [r for r in read(T1 / "phase1_strong.csv") if r["version"] == "seq"]
    return {"compute": med(rows, lambda r: r["dataset"], compute),
            "total": med(rows, lambda r: r["dataset"], lambda r: float(r["t_total"]))}


# ------------------------------------------------------------------ strong scaling
def strong(md: list[str]) -> None:
    base = serial_baseline()
    envs = {e: read(T2 / f"strong_{e}.csv") for e in ("host", "cluster")}
    envs = {e: r for e, r in envs.items() if r}
    if not envs or not base["compute"]:
        return
    omp = [r for r in read(T1 / "phase1_strong.csv") if r["version"] == "omp"]
    omp_c = med(omp, lambda r: (r["dataset"], int(r["threads_per_worker"])), compute)
    omp_t = med(omp, lambda r: (r["dataset"], int(r["threads_per_worker"])), lambda r: float(r["t_total"]))
    key = lambda r: (r["dataset"], int(r["workers"]))  # noqa: E731
    c = {e: med(r, key, compute) for e, r in envs.items()}
    t = {e: med(r, key, lambda r: float(r["t_total"])) for e, r in envs.items()}
    ld = {e: med(r, key, lambda r: float(r["t_load"])) for e, r in envs.items()}
    procs = sorted({int(r["workers"]) for r in next(iter(envs.values()))})
    sets = [s for s in ("100MB", "500MB", "full") if (s, 1) in next(iter(c.values()))]

    md.append("## Strong scaling (medians)\n")
    md.append("S = T_S / T_P with T_S the Phase 1 sequential time. Compute = pass 1 + Allreduce + "
              "pass 2 (wall clock on rank 0); total adds index distribution, opening the shards "
              "and writing the result files. E = S / P.\n")
    for s in sets:
        ts_c, ts_t = base["compute"][s], base["total"][s]
        md.append(f"\n**{s}**: T_S = {ts_c:.3f} s compute, {ts_t:.3f} s total\n")
        head = "| P | " + " | ".join(f"{e} compute (s) | {e} S | {e} E | {e} load (s) | {e} total (s) | {e} S total"
                                    for e in envs) + " | OpenMP S | OpenMP S total |"
        md.append(head)
        md.append("|---|" + "---|" * (6 * len(envs) + 2))
        for p in procs:
            cells = []
            for e in envs:
                cc, tt = c[e][(s, p)], t[e][(s, p)]
                cells += [f"{cc:.3f}", f"{ts_c / cc:.2f}", f"{ts_c / cc / p:.2f}", f"{ld[e][(s, p)]:.3f}",
                          f"{tt:.3f}", f"{ts_t / tt:.2f}"]
            cells += [f"{ts_c / omp_c[(s, p)]:.2f}", f"{ts_t / omp_t[(s, p)]:.2f}"] if (s, p) in omp_c else ["-", "-"]
            md.append(f"| {p} | " + " | ".join(cells) + " |")
    md.append("")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(procs, procs, ls="--", lw=1.2, color=MUTED, zorder=1)
    ax.annotate("ideal", (procs[-1], procs[-1]), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", color=INK2, fontsize=9)
    ts = base["compute"]["full"]
    series = [(f"MPI, {e}", [ts / c[e][("full", p)] for p in procs]) for e in envs]
    if ("full", 8) in omp_c:
        series.append(("OpenMP (Phase 1)", [ts / omp_c[("full", p)] for p in procs]))
    ends = []
    for color, (name, y) in zip(SERIES, series):
        ax.plot(procs, y, color=color, label=name, **MARK)
        ends.append([y[-1], f"{name}: {y[-1]:.2f}x"])
    end_labels(ax, ends, procs[-1])
    ax.set_xscale("log", base=2)
    ax.set_xticks(procs, [str(p) for p in procs])
    ax.set_xlabel("processes (MPI ranks) or threads (OpenMP)")
    ax.set_ylabel("speedup  T_seq / T_P  (compute)")
    ax.set_title("Strong scaling on the full set")
    ax.legend(loc="upper left")
    save(fig, "strong.png")


# ------------------------------------------------------------------ weak scaling
def weak(md: list[str]) -> None:
    envs = {e: read(T2 / f"weak_{e}.csv") for e in ("host", "cluster")}
    envs = {e: r for e, r in envs.items() if r}
    if not envs:
        return
    md.append("## Weak scaling (compute time, medians)\n")
    md.append("P ranks on the P x 700 MiB set. Scaled speedup = T_1(PW) / T_P(PW); "
              "weak efficiency = T_1(W) / T_P(PW).\n")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ends, ps = [], []
    for color, (e, rows) in zip(SERIES, envs.items()):
        m = med(rows, lambda r: (r["dataset"], int(r["workers"])), compute)
        sets = sorted({r["dataset"] for r in rows}, key=lambda s: int(s[:-2]))
        ps = [int(s[:-2]) // int(sets[0][:-2]) for s in sets]
        scaled = [m[(s, 1)] / m[(s, p)] for s, p in zip(sets, ps)]
        md.append(f"\n**{e}**\n")
        md.append("| P | set | T_1 (s) | T_P (s) | scaled speedup | weak efficiency |")
        md.append("|---|---|---|---|---|---|")
        for s, p, sc in zip(sets, ps, scaled):
            md.append(f"| {p} | {s} | {m[(s, 1)]:.3f} | {m[(s, p)]:.3f} | {sc:.2f} | "
                      f"{m[(sets[0], 1)] / m[(s, p)]:.2f} |")
        ax.plot(ps, scaled, color=color, label=f"MPI, {e}", **MARK)
        ends.append([scaled[-1], f"{e}: {scaled[-1]:.2f}x"])
    md.append("")
    ax.plot(ps, ps, ls="--", lw=1.2, color=MUTED, zorder=1)
    ax.annotate("ideal", (ps[-1], ps[-1]), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", color=INK2, fontsize=9)
    end_labels(ax, ends, ps[-1])
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps, [f"{p}\n{p * 700} MiB" for p in ps])
    ax.set_xlabel("processes P  /  problem size")
    ax.set_ylabel("scaled speedup  T_1(PW) / T_P(PW)")
    ax.set_title("Weak scaling: the dataset grows with P")
    ax.legend(loc="upper left")
    save(fig, "weak.png")


# ------------------------------------------------------------------ partition / hybrid
def partition(md: list[str]) -> None:
    rows = read(T2 / "partition_host.csv")
    if not rows:
        return
    tot = med(rows, lambda r: r["schedule"], lambda r: float(r["t_total"]))
    md.append("## Decomposition: what to balance (host, P = 8, full set)\n")
    md.append("Per-rank numbers are from the first measured run. Imbalance = (max - min) / max of "
              "each rank's own pass 1 + pass 2 time.\n")
    md.append("| partition | total (s) | records per rank (min-max) | MiB per rank (min-max) | "
              "rank time min (s) | rank time max (s) | imbalance |")
    md.append("|---|---|---|---|---|---|---|")
    for part in ("work", "count", "pixels"):
        rk = read(T2 / f"ranks_partition_{part}.csv")
        if not rk:
            continue
        tt = [float(r["t_pass1"]) + float(r["t_pass2"]) for r in rk]
        rec = [int(r["records"]) for r in rk]
        mib = [int(r["pixel_bytes"]) / 2**20 for r in rk]
        md.append(f"| {part} | {tot[f'part-{part}']:.3f} | {min(rec)}-{max(rec)} | "
                  f"{min(mib):.0f}-{max(mib):.0f} | {min(tt):.3f} | {max(tt):.3f} | "
                  f"{100 * (max(tt) - min(tt)) / max(tt):.1f}% |")
    md.append("")


def hybrid(md: list[str]) -> None:
    rows = read(T2 / "hybrid_host.csv")
    base = serial_baseline()
    if not rows or "full" not in base["compute"]:
        return
    key = lambda r: (int(r["workers"]), int(r["threads_per_worker"]))  # noqa: E731
    c, t = med(rows, key, compute), med(rows, key, lambda r: float(r["t_total"]))
    md.append("## Hybrid MPI x OpenMP (host, full set, 8 cores in every row)\n")
    md.append("| ranks x threads | compute (s) | S | total (s) | S total |")
    md.append("|---|---|---|---|---|")
    for k in sorted(c):
        md.append(f"| {k[0]} x {k[1]} | {c[k]:.3f} | {base['compute']['full'] / c[k]:.2f} | "
                  f"{t[k]:.3f} | {base['total']['full'] / t[k]:.2f} |")
    md.append("")


# ------------------------------------------------------------------ overlap
def overlap(md: list[str]) -> None:
    envs = {e: read(T2 / f"overlap_{e}.csv") for e in ("host", "cluster")}
    envs = {e: r for e, r in envs.items() if r}
    if not envs:
        return
    md.append("## Communication/computation overlap (500MB set, tensors shipped to rank 0)\n")
    md.append("Workers = P - 1 (rank 0 only collects). Comm = mean time a worker spends inside "
              "MPI_Send, or inside MPI_Wait + MPI_Isend, over the whole run (first measured run). "
              "Hidden = 1 - comm(isend) / comm(send).\n")
    fig, axes = plt.subplots(1, len(envs), figsize=(3.6 * len(envs), 3.6), squeeze=False, sharey=True)
    for ax, (e, rows) in zip(axes[0], envs.items()):
        p2 = med(rows, lambda r: (int(r["workers"]), r["schedule"]), lambda r: float(r["t_pass2"]))
        ps = sorted({int(r["workers"]) for r in rows})
        comm = {}
        for p in ps:
            for mode in ("send", "isend"):
                rk = [r for r in read(T2 / f"ranks_overlap_{e}_P={p}_{mode}.csv") if int(r["rank"]) > 0]
                comm[(p, mode)] = statistics.mean(float(r["t_comm"]) for r in rk) if rk else float("nan")
        md.append(f"\n**{e}**\n")
        md.append("| workers | pass 2 send (s) | pass 2 isend (s) | comm send (ms) | comm isend (ms) | hidden |")
        md.append("|---|---|---|---|---|---|")
        for p in ps:
            cs, ci = comm[(p, "send")], comm[(p, "isend")]
            md.append(f"| {p - 1} | {p2[(p, 'collect-send')]:.3f} | {p2[(p, 'collect-isend')]:.3f} | "
                      f"{cs * 1e3:.1f} | {ci * 1e3:.1f} | {100 * (1 - ci / cs):.0f}% |")
        x = np.arange(len(ps))
        for color, mode, dx in zip(SERIES, ("send", "isend"), (-0.1, 0.1)):
            ax.scatter(x + dx, [comm[(p, mode)] * 1e3 for p in ps], s=50, color=color,
                       edgecolors=SURFACE, linewidths=1.5, zorder=3,
                       label="MPI_Send (blocking)" if mode == "send" else "MPI_Isend (overlapped)")
        ax.set_yscale("log")
        ax.set_xticks(x, [str(p - 1) for p in ps])
        ax.set_xlabel("workers sending to one collector")
        ax.set_title(e, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("time per worker in send/wait (ms, log)")
    axes[0][0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Overlap hides communication only while the collector keeps up", x=0.01,
                 ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "overlap.png")
    md.append("")


# ------------------------------------------------------------------ ping-pong
def pingpong(md: list[str]) -> None:
    names = [("host", "host, shared memory"), ("same_node", "container, same node"),
             ("cross_node", "containers, across nodes (TCP)")]
    data = {label: read(T2 / f"pingpong_{n}.csv") for n, label in names}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    md.append("## Ping-pong: latency vs bandwidth\n")
    md.append("Latency = half the median round trip of messages up to 64 B. Bandwidth = best "
              "2 x bytes / RTT over all sizes.\n")
    md.append("| setting | one-way latency (us) | peak bandwidth (GB/s) | RTT 64 B (us) | RTT 64 KiB (us) | "
              "RTT 512 KiB (us) | RTT 64 MiB (ms) |")
    md.append("|---|---|---|---|---|---|---|")
    for color, (label, rows) in zip(SERIES, data.items()):
        x = np.array([int(r["bytes"]) for r in rows])
        y = np.array([float(r["rtt_median_us"]) for r in rows])
        ax.plot(x, y, color=color, label=label, lw=2, zorder=3)
        lat = float(np.median(y[x <= 64])) / 2
        bw = float(max(2 * x / y)) / 1e3
        at = lambda b: float(y[x == b][0])  # noqa: E731
        md.append(f"| {label} | {lat:.2f} | {bw:.1f} | {at(64):.2f} | {at(65536):.1f} | "
                  f"{at(524288):.1f} | {at(67108864) / 1e3:.1f} |")
    md.append("")
    for size, name in ((56, "Stats, 56 B"), (602112, "tensor, 588 KiB")):
        ax.axvline(size, color=MUTED, lw=1, ls="--", zorder=1)
        ax.annotate(name, (size, 0.03), xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", rotation=90, ha="left", va="bottom",
                    fontsize=8.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [1, 64, 4096, 262144, 16777216]
    ax.set_xticks(ticks, ["1 B", "64 B", "4 KiB", "256 KiB", "16 MiB"])
    ax.set_xlabel("message size")
    ax.set_ylabel("round-trip time (us, log scale)")
    ax.set_title("Ping-pong: latency-bound, then bandwidth-bound")
    ax.legend(loc="upper left")
    save(fig, "pingpong.png")


def main() -> int:
    if not T2.exists():
        print("no results/timings/phase2/: run scripts/bench/phase2.py first", file=sys.stderr)
        return 1
    md = ["# Phase 2 results\n", "Generated by plots/phase2.py from results/timings/phase2/.\n"]
    print("writing:")
    strong(md)
    weak(md)
    partition(md)
    hybrid(md)
    overlap(md)
    pingpong(md)
    (T2 / "summary.md").write_text("\n".join(md) + "\n")
    print("  results/timings/phase2/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
