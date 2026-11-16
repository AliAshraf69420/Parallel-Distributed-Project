#!/usr/bin/env python3
"""Phase 1 plots and tables from results/timings/phase1_*.csv (run scripts/bench/phase1.py first).

Writes to plots/phase1/:
  speedup.png      strong scaling: speedup S = T_seq / T_p vs threads, per set, with ideal line
  efficiency.png   E = S / p vs threads
  schedule.png     total time per OpenMP schedule (dot plot), per set x order, 4 and 8 threads
  weak.png         Gustafson: scaled speedup vs threads on the p x 700 MiB sets
  records.png      distribution of per-image pass-2 time (sequential, full set)
and results/timings/phase1_summary.md (all tables for the report, medians of the runs).

Usage: python plots/phase1.py
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
TIMINGS = REPO / "results" / "timings"
OUT = REPO / "plots" / "phase1"

# Reference palette (light mode), see the dataviz guidelines: first three categorical
# slots validate for all pairs; text uses ink colors, never the series color.
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


def read(name: str) -> list[dict]:
    path = TIMINGS / name
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def medians(rows: list[dict], key, field: str = "t_total") -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(float(r[field]))
    return {k: statistics.median(v) for k, v in groups.items()}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  plots/phase1/{name}")


def compute_time(r: dict) -> float:
    """Timed compute: pass 1 + merge + pass 2 (excludes loading and writing output files)."""
    return float(r["t_pass1"]) + float(r["t_merge"]) + float(r["t_pass2"])


# ------------------------------------------------------------------ strong scaling
def strong(md: list[str]) -> None:
    rows = read("phase1_strong.csv")
    if not rows:
        return
    for r in rows:
        r["t_compute"] = compute_time(r)
    sets = [s for s in ("100MB", "500MB", "full") if any(r["dataset"] == s for r in rows)]
    tot = medians(rows, lambda r: (r["version"], r["dataset"], int(r["threads_per_worker"])))
    cmp_ = medians(rows, lambda r: (r["version"], r["dataset"], int(r["threads_per_worker"])), "t_compute")
    p1 = medians(rows, lambda r: (r["version"], r["dataset"], int(r["threads_per_worker"])), "t_pass1")
    p2 = medians(rows, lambda r: (r["version"], r["dataset"], int(r["threads_per_worker"])), "t_pass2")
    threads = sorted({int(r["threads_per_worker"]) for r in rows if r["version"] == "omp"})

    md.append("## Strong scaling (OpenMP, schedule static, medians)\n")
    md.append("T_S = sequential time. S = T_S / T_p, E = S / p, "
              "f = (1/S - 1/p) / (1 - 1/p) (experimentally determined serial fraction). "
              "Compute = pass 1 + merge + pass 2; total also includes loading and output files.\n")
    for s in sets:
        ts_c, ts_t = cmp_[("seq", s, 1)], tot[("seq", s, 1)]
        md.append(f"\n**{s}**: T_S = {ts_c:.3f} s compute, {ts_t:.3f} s total\n")
        md.append("| threads | pass 1 (s) | pass 2 (s) | compute (s) | S | E | f | total (s) | S total |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for p in threads:
            tc = cmp_[("omp", s, p)]
            sp = ts_c / tc
            f = (1 / sp - 1 / p) / (1 - 1 / p) if p > 1 else float("nan")
            tag = f"{p} (SMT)" if p > 8 else str(p)
            md.append(f"| {tag} | {p1[('omp', s, p)]:.3f} | {p2[('omp', s, p)]:.3f} | {tc:.3f} | "
                      f"{sp:.2f} | {sp / p:.2f} | {'-' if p == 1 else f'{f:.4f}'} | "
                      f"{tot[('omp', s, p)]:.3f} | {ts_t / tot[('omp', s, p)]:.2f} |")
    md.append("")

    for kind in ("speedup", "efficiency"):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        if kind == "speedup":
            ax.plot(threads, threads, ls="--", lw=1.2, color=MUTED, zorder=1)
            ax.annotate("ideal (S = p)", (threads[-1], threads[-1]), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center", color=INK2, fontsize=9)
        else:
            ax.axhline(1.0, ls="--", lw=1.2, color=MUTED, zorder=1)
        for color, s in zip(SERIES, sets):
            ts = cmp_[("seq", s, 1)]
            y = [ts / cmp_[("omp", s, p)] / (1 if kind == "speedup" else p) for p in threads]
            ax.plot(threads, y, marker="o", color=color, label=s, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f"{s}  {y[threads.index(8)] if 8 in threads else y[-1]:.2f}"
                        f"{'x' if kind == 'speedup' else ''} @8",
                        (8, y[threads.index(8)]) if 8 in threads else (threads[-1], y[-1]),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads, [f"{p}\n(SMT)" if p > 8 else str(p) for p in threads])
        ax.set_xlabel("OpenMP threads (1 per physical core up to 8)")
        if kind == "speedup":
            ax.set_ylabel("speedup  T_seq / T_p  (compute)")
            ax.set_title("Strong scaling: speedup vs threads")
        else:
            ax.set_ylabel("efficiency  S / p")
            ax.set_ylim(0, 1.15)
            ax.set_title("Strong scaling: parallel efficiency")
        ax.legend(loc="upper left")
        save(fig, f"{kind}.png")


# ------------------------------------------------------------------ scheduling
def schedule(md: list[str]) -> None:
    rows = read("phase1_schedule.csv")
    if not rows:
        return
    med = medians(rows, lambda r: (r["dataset"], r["order"], int(r["threads_per_worker"]), r["schedule"]))
    p2 = medians(rows, lambda r: (r["dataset"], r["order"], int(r["threads_per_worker"]), r["schedule"]), "t_pass2")
    scheds = list(dict.fromkeys(r["schedule"] for r in rows))
    panels = list(dict.fromkeys((r["dataset"], r["order"]) for r in rows))
    threads = sorted({int(r["threads_per_worker"]) for r in rows})

    md.append("## OpenMP scheduling (total time, medians; best per row in bold)\n")
    md.append("| set | order | threads | " + " | ".join(scheds) + " |")
    md.append("|---|---|---|" + "---|" * len(scheds))
    for s, o in panels:
        for t in threads:
            vals = [med[(s, o, t, k)] for k in scheds]
            best = min(vals)
            cells = [f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in vals]
            md.append(f"| {s} | {o} | {t} | " + " | ".join(cells) + " |")
    md.append("\nPass 2 only (s):\n")
    md.append("| set | order | threads | " + " | ".join(scheds) + " |")
    md.append("|---|---|---|" + "---|" * len(scheds))
    for s, o in panels:
        for t in threads:
            md.append(f"| {s} | {o} | {t} | " + " | ".join(f"{p2[(s, o, t, k)]:.3f}" for k in scheds) + " |")
    md.append("")

    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(scheds))
    for ax, (s, o) in zip(axes, panels):
        for color, t, dy in zip(SERIES, threads, (-0.12, 0.12)):
            ax.scatter([med[(s, o, t, k)] for k in scheds], y + dy, s=46, color=color,
                       edgecolors=SURFACE, linewidths=1.5, zorder=3, label=f"{t} threads")
        ax.set_title(f"{s}, order={o}", fontsize=10)
        ax.set_yticks(y, [k.replace(";", ",") for k in scheds])
        ax.invert_yaxis()
        hi = max(med[(s, o, t, k)] for t in threads for k in scheds)
        ax.set_xlim(0, hi * 1.15)
        ax.set_xlabel("total time (s)")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("OpenMP schedule comparison (lower is better)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "schedule.png")


# ------------------------------------------------------------------ weak scaling
def weak(md: list[str]) -> None:
    rows = read("phase1_weak.csv")
    if not rows:
        return
    for r in rows:
        r["t_compute"] = compute_time(r)
    med = medians(rows, lambda r: (r["dataset"], int(r["threads_per_worker"])), "t_compute")
    sets = sorted({r["dataset"] for r in rows}, key=lambda s: int(s[:-2]))
    ps = [int(s[:-2]) // int(sets[0][:-2]) for s in sets]
    t1_base = med[(sets[0], 1)]
    scaled = [med[(s, 1)] / med[(s, p)] for s, p in zip(sets, ps)]       # T1(pW) / Tp(pW)
    weak_eff = [t1_base / med[(s, p)] for s, p in zip(sets, ps)]         # T1(W) / Tp(pW)
    # Gustafson: S(p) = p - s (p - 1); fit the serial share s from the p = 8 point.
    p8 = ps[-1]
    s_fit = (p8 - scaled[-1]) / (p8 - 1)

    md.append("## Weak scaling / Gustafson (compute time, medians)\n")
    md.append("Each thread gets 700 MiB of decoded pixels. Scaled speedup = T_1(pW) / T_p(pW); "
              "weak efficiency = T_1(W) / T_p(pW). Gustafson: S(p) = p - s(p - 1).\n")
    md.append("| p | set | T_1 (s) | T_p (s) | scaled speedup | weak efficiency |")
    md.append("|---|---|---|---|---|---|")
    for s, p, sc, we in zip(sets, ps, scaled, weak_eff):
        md.append(f"| {p} | {s} | {med[(s, 1)]:.3f} | {med[(s, p)]:.3f} | {sc:.2f} | {we:.2f} |")
    md.append(f"\nFitted Gustafson serial share s = {s_fit:.4f} (from p = {p8}).\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ps, ps, ls="--", lw=1.2, color=MUTED, zorder=1)
    ax.annotate("ideal", (ps[-1], ps[-1]), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", color=INK2, fontsize=9)
    pp = np.linspace(1, ps[-1], 50)
    ax.plot(pp, pp - s_fit * (pp - 1), lw=1.2, color=SERIES[1], zorder=2,
            label=f"Gustafson fit, s = {s_fit:.3f}")
    ax.plot(ps, scaled, marker="o", color=SERIES[0], zorder=3, markeredgecolor=SURFACE,
            markeredgewidth=2, label="measured scaled speedup")
    ax.annotate(f"{scaled[-1]:.2f}x", (ps[-1], scaled[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ps, [f"{p}\n{s}" for p, s in zip(ps, sets)])
    ax.set_xlabel("threads p  /  problem size (p x 700 MiB)")
    ax.set_ylabel("scaled speedup  T_1(pW) / T_p(pW)")
    ax.set_title("Weak scaling (Gustafson)")
    ax.legend(loc="upper left")
    save(fig, "weak.png")


# ------------------------------------------------------------------ per-image work
def records(md: list[str]) -> None:
    rows = read("phase1_records_full.csv")
    if not rows:
        return
    t = np.array([float(r["pass2_us"]) for r in rows])
    area = np.array([int(r["bw"]) * int(r["bh"]) for r in rows])
    pct = {q: float(np.percentile(t, q)) for q in (1, 5, 50, 95, 99)}
    corr = float(np.corrcoef(t, area)[0, 1])
    md.append("## Work per image (sequential pass 2, full set)\n")
    md.append(f"median {pct[50]:.0f} us, p5 {pct[5]:.0f}, p95 {pct[95]:.0f}, p99 {pct[99]:.0f}, "
              f"max {t.max():.0f}; p95/p5 = {pct[95] / pct[5]:.2f}; "
              f"coefficient of variation {t.std() / t.mean():.3f}; "
              f"correlation with crop area {corr:.2f} "
              f"(crop areas span p95/p5 = {np.percentile(area, 95) / np.percentile(area, 5):.1f}x).\n")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    hi = np.percentile(t, 99.5)
    ax.hist(np.clip(t, None, hi), bins=80, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(pct[50], color=INK2, lw=1.2, ls="--")
    ax.annotate(f"median {pct[50]:.0f} us", (pct[50], ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlabel("pass-2 time per image (us), clipped at p99.5")
    ax.set_ylabel("images")
    ax.set_title("Work per image is nearly uniform")
    ax.grid(axis="x", visible=False)
    save(fig, "records.png")


def main() -> int:
    if not TIMINGS.exists():
        print("no results/timings/: run scripts/bench/phase1.py first", file=sys.stderr)
        return 1
    md = ["# Phase 1 results\n", "Generated by plots/phase1.py from results/timings/phase1_*.csv.\n"]
    print("writing:")
    strong(md)
    schedule(md)
    weak(md)
    records(md)
    (TIMINGS / "phase1_summary.md").write_text("\n".join(md) + "\n")
    print("  results/timings/phase1_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
