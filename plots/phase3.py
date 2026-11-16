#!/usr/bin/env python3
"""Phase 3 plots and tables from results/timings/phase3/ (run scripts/bench/phase3.py first).

Writes to plots/phase3/:
  compare.png   run time vs input size for Serial, OpenMP, MPI and Hadoop (log-log)
  failure.png   the failure experiment over time: map progress and under-replicated blocks
and results/timings/phase3/summary.md (all tables for the report, medians of the runs),
including the cross-model comparison table and the estimate of when Hadoop would win.

Usage: python plots/phase3.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
T3 = REPO / "results" / "timings" / "phase3"
T2 = REPO / "results" / "timings" / "phase2"
T1 = REPO / "results" / "timings"
OUT = REPO / "plots" / "phase3"
SETS = ["100MB", "500MB", "full"]

# Reference palette (light mode), categorical slots 1-4 in fixed order; ink for all text.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
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


def med(rows, key, value) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(value(r))
    return {k: statistics.median(v) for k, v in groups.items()}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  plots/phase3/{name}")


def pixel_mib() -> dict:
    """Decoded size of each set in MiB (what the binary versions process)."""
    out = {}
    for s in SETS:
        meta = REPO / "data" / "processed" / s / "meta.json"
        if meta.exists():
            out[s] = json.loads(meta.read_text())["pixel_bytes"] / 2**20
    return out or {"100MB": 100.4, "500MB": 500.5, "full": 6046.1}


# ------------------------------------------------------------------ HDFS
def hdfs(md: list[str]) -> None:
    path = T3 / "hdfs.json"
    if not path.exists():
        return
    h = json.loads(path.read_text())
    md.append("## HDFS inspection (`hdfs fsck -files -blocks -locations`)\n")
    md.append("Block size 128 MiB, default replication 2, 3 DataNodes.\n")
    md.append("| set | text size (MiB) | upload (s) | MB/s | blocks | avg replication | stored with replicas (MiB) | "
              "replicas per DataNode | status |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in SETS:
        if s not in h["fsck"]:
            continue
        f, u = h["fsck"][s], h["upload"][s]
        md.append(f"| {s} | {f['file_bytes'] / 2**20:.0f} | {u['seconds']:.1f} | {u['mb_per_s']:.0f} | "
                  f"{f['total_blocks']} | {f['average_replication']:.1f} | "
                  f"{f['bytes_with_replicas'] / 2**20:.0f} | "
                  f"{' / '.join(str(v) for v in f['replicas_per_datanode'].values())} | {f['status']} |")
    md.append("\n**Changing the replication factor** of the 500MB file (`hdfs dfs -setrep -w N`):\n")
    md.append("| replication | time to apply (s) | blocks | avg replication | stored (MiB) | replicas per DataNode | "
              "under / over-replicated |")
    md.append("|---|---|---|---|---|---|---|")
    for e in h["replication"]:
        md.append(f"| {e['replication']} | {e['seconds_to_apply']:.1f} | {e['total_blocks']} | "
                  f"{e['average_replication']:.1f} | {e['bytes_with_replicas'] / 2**20:.0f} | "
                  f"{' / '.join(str(v) for v in e['replicas_per_datanode'].values())} | "
                  f"{e['under_replicated']} / {e['over_replicated']} |")
    md.append("")


# ------------------------------------------------------------------ comparison
def compare(md: list[str]) -> None:
    jobs = read(T3 / "jobs.csv")
    cmp_rows = [r for r in jobs if r["experiment"] == "compare"]
    if not cmp_rows:
        return
    el = med(cmp_rows, lambda r: (r["job"], r["set"]), lambda r: float(r["elapsed_s"]))
    cpu = med(cmp_rows, lambda r: (r["job"], r["set"]), lambda r: float(r["cpu_ms"]) / 1e3)
    maps = med(cmp_rows, lambda r: (r["job"], r["set"]), lambda r: float(r["launched_maps"]))
    p1 = read(T1 / "phase1_strong.csv")
    tot = lambda rows, v, t: med([r for r in rows if r["version"] == v and int(r["threads_per_worker"]) == t],  # noqa: E731
                                 lambda r: r["dataset"], lambda r: float(r["t_total"]))
    serial, omp = tot(p1, "seq", 1), tot(p1, "omp", 8)
    p1only = med([r for r in p1 if r["version"] == "seq"], lambda r: r["dataset"], lambda r: float(r["t_pass1"]))
    mpi = med([r for r in read(T2 / "strong_host.csv") if int(r["workers"]) == 8],
              lambda r: r["dataset"], lambda r: float(r["t_total"]))
    size = pixel_mib()
    sets = [s for s in SETS if ("stats", s) in el]

    md.append("## Cross-model comparison (end-to-end run time in seconds, medians)\n")
    md.append("Serial, OpenMP (8 threads) and MPI (8 ranks, one machine) read the binary sets; their "
              "times are from Phases 1 and 2. Hadoop reads the same images as base64 text from HDFS. "
              "Hadoop pipeline = Job 1 (statistics) + Job 2 (normalize), i.e. the same two passes. "
              "Uploading to HDFS is not included.\n")
    md.append("| input | Serial | OpenMP | MPI | Hadoop Job 1 | Hadoop Job 2 | Hadoop pipeline | Hadoop / Serial | "
              "map tasks (Job 1) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    hp = {}
    for s in sets:
        hp[s] = el[("stats", s)] + el.get(("normalize", s), 0.0)
        md.append(f"| {s} ({size[s]:.0f} MiB) | {serial.get(s, float('nan')):.2f} | {omp.get(s, float('nan')):.2f} | "
                  f"{mpi.get(s, float('nan')):.2f} | {el[('stats', s)]:.1f} | {el.get(('normalize', s), float('nan')):.1f} | "
                  f"{hp[s]:.1f} | {hp[s] / serial[s]:.0f}x | {maps[('stats', s)]:.0f} |")
    md.append("")

    # When would Hadoop win? Fixed cost and per-MiB cost from the smallest and largest set.
    a, b = sets[0], sets[-1]
    per_mib = (hp[b] - hp[a]) / (size[b] - size[a])
    fixed = hp[a] - per_mib * size[a]
    serial_per_mib = serial[b] / size[b]
    cpu_total = cpu[("stats", b)] + cpu.get(("normalize", b), 0.0)
    md.append("**When would Hadoop pay off?**\n")
    md.append(f"- Fixed cost per pipeline (two jobs: JVM start, YARN scheduling, container launch): "
              f"about {fixed:.0f} s, measured as the intercept of a line through the {a} and {b} runs.")
    md.append(f"- Marginal cost on this cluster: {per_mib * 1e3:.1f} ms per MiB of pixels, against "
              f"{serial_per_mib * 1e3:.2f} ms per MiB for the serial program. "
              + ("Hadoop's marginal cost is higher, so on this cluster it never catches up: more data only "
                 "widens the gap." if per_mib >= serial_per_mib else
                 f"The curves cross near {fixed / (serial_per_mib - per_mib) / 1024:.0f} GiB of pixels."))
    md.append(f"- Hadoop used {cpu_total:.0f} s of CPU time for the full set (both jobs), "
              f"{cpu_total / serial[b]:.0f} times the serial program's {serial[b]:.1f} s: decoding base64 text, "
              f"JVM record handling and the streaming pipes cost more than the computation.")
    md.append(f"- With that CPU cost and ideal scaling, Hadoop needs about {cpu_total / serial[b]:.0f} busy cores "
              f"to match the serial program and about {cpu_total / mpi[b]:.0f} to match 8 MPI ranks, before "
              f"counting the fixed {fixed:.0f} s. It wins when the data no longer fits one machine "
              f"(disk, or a run time long enough that a node failure becomes likely), not on speed.\n")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    x = [size[s] for s in sets]
    series = [("Hadoop (Job 1 + Job 2)", [hp[s] for s in sets]), ("Serial", [serial[s] for s in sets]),
              ("OpenMP, 8 threads", [omp[s] for s in sets]), ("MPI, 8 ranks", [mpi[s] for s in sets])]
    ends = []
    for color, (name, y) in zip(SERIES, series):
        ax.plot(x, y, color=color, label=name, **MARK)
        ends.append([math.log10(y[-1]), f"{name}: {y[-1]:.1f} s"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Direct labels at the line ends, nudged apart (in log space) so they never overlap.
    lo, hi = (math.log10(v) for v in ax.get_ylim())
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.065 * (hi - lo))
    for ly, text in ends:
        ax.annotate(text, (x[-1], 10 ** ly), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2, annotation_clip=False)
    ax.set_xticks(x, [f"{s}\n{size[s]:.0f} MiB" for s in sets])
    ax.minorticks_off()
    ax.set_xlabel("input (decoded pixels)")
    ax.set_ylabel("end-to-end run time (s, log scale)")
    ax.set_title("Hadoop pays a fixed start-up cost and a higher cost per byte")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "compare.png")


# ------------------------------------------------------------------ combiner
def combiner(md: list[str]) -> None:
    jobs = [r for r in read(T3 / "jobs.csv") if r["job"] == "stats" and r["experiment"] in ("compare", "combiner")]
    if not any(r["experiment"] == "combiner" for r in jobs):
        return
    key = lambda r: (r["set"], int(r["combiner"]))  # noqa: E731
    cols = ["map_output_records", "combine_input_records", "combine_output_records", "reduce_input_records",
            "reduce_shuffle_bytes", "launched_maps"]
    m = {c: med(jobs, key, lambda r, c=c: float(r[c])) for c in cols}
    el = med(jobs, key, lambda r: float(r["elapsed_s"]))
    md.append("## Combiner experiment (Job 1, medians)\n")
    md.append("| set | combiner | map tasks | map output records | combine input | combine output | "
              "reduce input records | shuffle bytes | time (s) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in ("500MB", "full"):
        for c in (1, 0):
            if (s, c) not in el:
                continue
            md.append(f"| {s} | {'yes' if c else 'no'} | {m['launched_maps'][(s, c)]:.0f} | "
                      f"{m['map_output_records'][(s, c)]:.0f} | {m['combine_input_records'][(s, c)]:.0f} | "
                      f"{m['combine_output_records'][(s, c)]:.0f} | {m['reduce_input_records'][(s, c)]:.0f} | "
                      f"{m['reduce_shuffle_bytes'][(s, c)]:.0f} | {el[(s, c)]:.1f} |")
    for s in ("500MB", "full"):
        if (s, 1) in el and (s, 0) in el:
            a, b = m["reduce_shuffle_bytes"][(s, 0)], m["reduce_shuffle_bytes"][(s, 1)]
            md.append(f"\n{s}: the combiner cuts the shuffle from {a / 1024:.0f} KiB to {b / 1024:.0f} KiB "
                      f"({a / b:.0f} times less network traffic) and the reducer's input from "
                      f"{m['reduce_input_records'][(s, 0)]:.0f} to {m['reduce_input_records'][(s, 1)]:.0f} records.")
    md.append("")


# ------------------------------------------------------------------ failure
def failure(md: list[str]) -> None:
    path = T3 / "failure.json"
    if not path.exists():
        return
    f = json.loads(path.read_text())
    job = f["job"]
    normal = med([r for r in read(T3 / "jobs.csv") if r["experiment"] == "compare" and r["job"] == "stats"
                  and r["set"] == "full"], lambda r: 0, lambda r: float(r["elapsed_s"])).get(0, float("nan"))
    nmaps = med([r for r in read(T3 / "jobs.csv") if r["experiment"] == "compare" and r["job"] == "stats"
                 and r["set"] == "full"], lambda r: 0, lambda r: float(r["launched_maps"])).get(0, float("nan"))
    md.append("## Failure experiment (Job 1 on the full set)\n")
    md.append(f"At t = {f['killed_at_s']:.0f} s (maps at {f['map_progress_at_kill']} %) one worker was stopped: "
              f"{f['victim']}. The ApplicationMaster was running on {f.get('am_host') or 'another node'}.\n")
    md.append("| | value |")
    md.append("|---|---|")
    md.append(f"| job finished | {'yes' if f['job_exit_code'] == 0 else 'NO'} (exit code {f['job_exit_code']}) |")
    md.append(f"| result | {f['check_output']} |")
    md.append(f"| run time | {job['elapsed_s']:.1f} s (normal run: {normal:.1f} s) |")
    md.append(f"| map task attempts launched | {job['launched_maps']} (normal run: {nmaps:.0f}) |")
    md.append(f"| failed map attempts | {job['failed_maps']} |")
    md.append(f"| killed map attempts | {job['killed_maps']} |")
    md.append(f"| attempt failures reported to the client | {len(f['task_attempt_failures'])} |")
    md.append(f"| NameNode noticed the dead DataNode | {f['detected_after_kill_s']} s after the stop |")
    md.append(f"| all blocks re-replicated | {f['rereplicated_after_kill_s']} s after the stop |")
    md.append("\n`hdfs fsck /cub/full.txt` at four moments:\n")
    md.append("| moment | DataNodes | blocks | under-replicated | missing replicas | avg replication | status |")
    md.append("|---|---|---|---|---|---|---|")
    for label, k in (("before", "fsck_before"), ("after the node died", "fsck_after_death"),
                     ("after re-replication", "fsck_after_rereplication"), ("after the worker rejoined", "fsck_after_rejoin")):
        if k in f:
            e = f[k]
            md.append(f"| {label} | {e['datanodes']} | {e['total_blocks']} | {e['under_replicated']} | "
                      f"{e['missing_replicas']} | {e['average_replication']:.2f} | {e['status']} |")
    md.append("")

    tl = f["timeline"]
    if not tl:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True, height_ratios=[1, 1])
    t = [e["t"] for e in tl]
    a1.plot(t, [max(e["map_progress"], 0) for e in tl], color=SERIES[0], **MARK)
    a1.set_ylabel("map progress (%)")
    a1.set_ylim(0, 105)
    a1.set_title("Failure experiment: one worker stopped during the job")
    a2.plot(t, [e["under_replicated"] for e in tl], color=SERIES[1], **MARK)
    a2.set_ylabel("under-replicated blocks")
    a2.set_xlabel("seconds since the job was submitted")
    a2.set_xlim(0, None)
    for ax in (a1, a2):
        ax.axvline(f["killed_at_s"], color=MUTED, lw=1, ls="--", zorder=1)
    a1.annotate("worker stopped", (f["killed_at_s"], 4), xytext=(5, 0), textcoords="offset points",
                fontsize=8.5, color=INK2)
    if job.get("elapsed_s"):
        for ax in (a1, a2):
            ax.axvline(job["elapsed_s"], color=MUTED, lw=1, ls=":", zorder=1)
        a1.annotate("job finished", (job["elapsed_s"], 4), xytext=(5, 0), textcoords="offset points",
                    fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "failure.png")


def main() -> int:
    if not T3.exists():
        print("no results/timings/phase3/: run scripts/bench/phase3.py first", file=sys.stderr)
        return 1
    md = ["# Phase 3 results\n", "Generated by plots/phase3.py from results/timings/phase3/.\n"]
    print("writing:")
    hdfs(md)
    compare(md)
    combiner(md)
    failure(md)
    (T3 / "summary.md").write_text("\n".join(md) + "\n")
    print("  results/timings/phase3/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
