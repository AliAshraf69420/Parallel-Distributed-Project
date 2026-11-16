#!/usr/bin/env python3
"""Phase 3 experiment runner (Hadoop Streaming on the Docker cluster).

Needs: `make`, the text sets (preprocessing/to_text.py), the cluster up
(scripts/hadoop/cluster.sh up) and the conda env active.

Experiments (select with --only; default: all, in this order)
  hdfs       upload the three sets (timed); `hdfs fsck` each one (blocks, replication,
             DataNodes per block); change the replication factor of the 500MB set
             2 -> 3 -> 1 -> 2 and record what fsck and `hdfs dfs -du` show each time
  compare    Job 1 (statistics) + Job 2 (normalize) on 100MB / 500MB / full, --runs times;
             the first run of every set is checked bit-exactly against the reference
  combiner   Job 1 without -combiner on 500MB and full (the runs with a combiner come
             from `compare`): job counters and time
  failure    start Job 1 on the full set; when the maps reach 25 %, stop one worker (its
             NodeManager and its DataNode, never the node running the ApplicationMaster);
             record failed/killed/relaunched task attempts, whether the result is still
             correct, and fsck before / after the failure / after re-replication

Output: results/timings/phase3/jobs.csv (one row per job run, with its counters),
        hdfs.json, failure.json, fsck_*.txt, logs/*.log, env.json

Usage (from the repo root):
  python scripts/bench/phase3.py [--only hdfs compare combiner failure] [--runs 3]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "timings" / "phase3"
LOGS = OUT / "logs"
CLUSTER = "scripts/hadoop/cluster.sh"
JOB = "scripts/hadoop/job.sh"
SETS = ["100MB", "500MB", "full"]
ENV = {**os.environ, "PYTHON": sys.executable}

COUNTERS = {
    "launched_maps": "Launched map tasks",
    "launched_reduces": "Launched reduce tasks",
    "failed_maps": "Failed map tasks",
    "killed_maps": "Killed map tasks",
    "map_input_records": "Map input records",
    "map_output_records": "Map output records",
    "map_output_bytes": "Map output bytes",
    "combine_input_records": "Combine input records",
    "combine_output_records": "Combine output records",
    "reduce_shuffle_bytes": "Reduce shuffle bytes",
    "reduce_input_records": "Reduce input records",
    "reduce_output_records": "Reduce output records",
    "map_task_ms": "Total time spent by all map tasks (ms)",
    "reduce_task_ms": "Total time spent by all reduce tasks (ms)",
    "cpu_ms": "CPU time spent (ms)",
    "hdfs_bytes_read": "HDFS: Number of bytes read",
}
FIELDS = ["experiment", "job", "set", "tag", "run", "combiner", "elapsed_s", *COUNTERS]


def sh(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True)
    if check and res.returncode != 0:
        sys.exit(f"command failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}{res.stderr}")
    return res


def hdfs(*args: str, check: bool = True) -> str:
    return sh([CLUSTER, "exec", *args], check=check).stdout


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    row = {}
    for key, name in COUNTERS.items():
        m = re.search(r"^\s*" + re.escape(name) + r"=(\d+)", text, re.M)
        row[key] = int(m.group(1)) if m else 0
    m = re.search(r"elapsed_s=([\d.]+)", text)
    row["elapsed_s"] = round(float(m.group(1)), 2) if m else float("nan")
    return row


def record(row: dict) -> None:
    path = OUT / "jobs.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def run_job(experiment: str, job: str, set_name: str, tag: str, run: int, combiner: bool = True,
            extra: list[str] | None = None) -> dict:
    cmd = [JOB, job, set_name, "--tag", tag, *(extra or [])]
    if job == "stats" and not combiner:
        cmd.append("--no-combiner")
    t0 = time.perf_counter()
    sh(cmd)
    row = parse_log(LOGS / f"{job}-{set_name}-{tag}.log")
    row.update(experiment=experiment, job=job, set=set_name, tag=tag, run=run,
               combiner=int(combiner) if job == "stats" else "")
    record(row)
    print(f"  {job:<9} {set_name:<6} {tag:<10} {row['elapsed_s']:7.1f} s  "
          f"(maps {row['launched_maps']}, wall incl. fetch {time.perf_counter() - t0:.1f} s)", flush=True)
    return row


def fsck(path: str, save_as: str) -> dict:
    text = hdfs("hdfs", "fsck", path, "-files", "-blocks", "-locations", check=False)
    (OUT / save_as).write_text(text)

    def num(pattern: str, cast=int, default=0):
        m = re.search(pattern, text, re.M)
        return cast(m.group(1)) if m else default
    per_dn: dict[str, int] = {}
    for ip in re.findall(r"DatanodeInfoWithStorage\[([\d.]+):", text):
        per_dn[ip] = per_dn.get(ip, 0) + 1
    return {
        "status": "HEALTHY" if "is HEALTHY" in text else ("CORRUPT" if "is CORRUPT" in text else "UNKNOWN"),
        "total_size_bytes": num(r"Total size:\s+(\d+) B"),
        "total_blocks": num(r"Total blocks \(validated\):\s+(\d+)"),
        "minimally_replicated": num(r"Minimally replicated blocks:\s+(\d+)"),
        "under_replicated": num(r"Under-replicated blocks:\s+(\d+)"),
        "over_replicated": num(r"Over-replicated blocks:\s+(\d+)"),
        "missing_replicas": num(r"Missing replicas:\s+(\d+)"),
        "default_replication": num(r"Default replication factor:\s+(\d+)"),
        "average_replication": num(r"Average block replication:\s+([\d.]+)", float, 0.0),
        "datanodes": num(r"Number of data-nodes:\s+(\d+)"),
        "replicas_per_datanode": dict(sorted(per_dn.items())),
    }


def du(path: str) -> dict:
    parts = hdfs("hdfs", "dfs", "-du", path).split()
    return {"file_bytes": int(parts[0]), "bytes_with_replicas": int(parts[1])}


# ------------------------------------------------------------------ experiments
def exp_hdfs() -> None:
    print("HDFS: upload, fsck, replication")
    result: dict = {"upload": {}, "fsck": {}, "replication": []}
    for s in SETS:
        hdfs("hdfs", "dfs", "-rm", "-f", "-skipTrash", f"/cub/{s}.txt", check=False)
        t0 = time.perf_counter()
        sh([JOB, "put", s])
        dt = time.perf_counter() - t0
        size = (REPO / "data" / "processed" / "text" / f"{s}.txt").stat().st_size
        result["upload"][s] = {"seconds": round(dt, 2), "bytes": size,
                               "mb_per_s": round(size / dt / 1e6, 1)}
        result["fsck"][s] = {**fsck(f"/cub/{s}.txt", f"fsck_{s}.txt"), **du(f"/cub/{s}.txt")}
        f = result["fsck"][s]
        print(f"  {s}: uploaded in {dt:.1f} s; {f['total_blocks']} blocks, replication "
              f"{f['average_replication']}, {f['datanodes']} DataNodes, {f['replicas_per_datanode']}")
    path = "/cub/500MB.txt"
    for rep in (3, 1, 2):
        t0 = time.perf_counter()
        hdfs("hdfs", "dfs", "-setrep", "-w", str(rep), path)
        dt = time.perf_counter() - t0
        time.sleep(8)                       # let excess replicas be deleted before looking
        entry = {"replication": rep, "seconds_to_apply": round(dt, 1),
                 **fsck(path, f"fsck_500MB_rep{rep}.txt"), **du(path)}
        result["replication"].append(entry)
        print(f"  setrep {rep}: {dt:.1f} s; avg replication {entry['average_replication']}, "
              f"{entry['bytes_with_replicas'] / 2**20:.0f} MiB stored, {entry['replicas_per_datanode']}")
    (OUT / "hdfs.json").write_text(json.dumps(result, indent=2))


def exp_compare(runs: int) -> None:
    print("comparison runs (Job 1 + Job 2)")
    for s in SETS:
        for k in range(runs):
            run_job("compare", "stats", s, f"cmp{k}", k)
            run_job("compare", "normalize", s, f"cmp{k}", k)
            if k == 0:
                chk = sh([JOB, "check", s], check=False)
                if chk.returncode != 0:
                    sys.exit(f"correctness check failed for {s}:\n{chk.stdout}{chk.stderr}")
                print("  " + chk.stdout.strip().splitlines()[-1])


def exp_combiner(runs: int) -> None:
    print("combiner experiment (Job 1 without -combiner)")
    for s in ("500MB", "full"):
        for k in range(runs):
            run_job("combiner", "stats", s, f"nocomb{k}", k, combiner=False)
        chk = sh([JOB, "check", s], check=False)
        if chk.returncode != 0:
            sys.exit(f"result without combiner is wrong for {s}:\n{chk.stdout}{chk.stderr}")
        print("  " + chk.stdout.strip().splitlines()[-1])


def am_host() -> str:
    """Host of the running application's ApplicationMaster, from the ResourceManager REST API."""
    try:
        with urllib.request.urlopen("http://localhost:8088/ws/v1/cluster/apps?states=RUNNING", timeout=5) as r:
            apps = (json.load(r).get("apps") or {}).get("app") or []
        return apps[0]["amHostHttpAddress"].split(":")[0] if apps else ""
    except Exception:  # noqa: BLE001
        return ""


def exp_failure() -> None:
    print("failure experiment (Job 1 on the full set, one worker stopped mid-run)")
    path = "/cub/full.txt"
    result: dict = {"set": "full", "timeline": []}
    result["fsck_before"] = fsck(path, "fsck_failure_before.txt")
    log = LOGS / "stats-full-failure.log"
    if log.exists():
        log.unlink()
    t0 = time.perf_counter()
    proc = subprocess.Popen([JOB, "stats", "full", "--tag", "failure"], cwd=REPO, env=ENV,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def progress() -> int:
        if not log.exists():
            return -1
        m = re.findall(r" map (\d+)% reduce", log.read_text(errors="replace"))
        return int(m[-1]) if m else -1

    while proc.poll() is None and progress() < 25:
        time.sleep(1)
    if proc.poll() is not None:
        sys.exit("job finished before the failure could be injected")
    am = am_host()
    victim = next(k for k in (2, 3, 1) if f"nm{k}" != am)
    t_kill = time.perf_counter() - t0
    sh([CLUSTER, "stop", f"nm{victim}"])
    sh([CLUSTER, "stop", f"dn{victim}"])
    result.update(am_host=am, victim=f"nm{victim} + dn{victim}", killed_at_s=round(t_kill, 1),
                  map_progress_at_kill=progress())
    print(f"  t={t_kill:.0f} s, map {progress()} %: stopped nm{victim} and dn{victim} (AM on {am or '?'})")

    # Watch HDFS until the lost replicas have been re-created on the two remaining DataNodes.
    seen_under, t_detect, t_healed = False, None, None
    while time.perf_counter() - t0 < 900:
        f = fsck(path, "fsck_failure_latest.txt")
        t = time.perf_counter() - t0
        result["timeline"].append({"t": round(t, 1), "map_progress": progress(),
                                   "job_running": proc.poll() is None, "datanodes": f["datanodes"],
                                   "under_replicated": f["under_replicated"],
                                   "missing_replicas": f["missing_replicas"], "status": f["status"]})
        if f["under_replicated"] > 0 and not seen_under:
            seen_under, t_detect = True, t
            (OUT / "fsck_failure_after_death.txt").write_text((OUT / "fsck_failure_latest.txt").read_text())
            result["fsck_after_death"] = f
            print(f"  t={t:.0f} s: NameNode sees {f['datanodes']} DataNodes, "
                  f"{f['under_replicated']} under-replicated blocks")
        if seen_under and f["under_replicated"] == 0 and t_healed is None:
            t_healed = t
            result["fsck_after_rereplication"] = f
            (OUT / "fsck_failure_after_rereplication.txt").write_text(
                (OUT / "fsck_failure_latest.txt").read_text())
            print(f"  t={t:.0f} s: all blocks back at full replication on {f['datanodes']} DataNodes")
        if proc.poll() is not None and (t_healed is not None or not seen_under):
            break
        time.sleep(4)
    (OUT / "fsck_failure_latest.txt").unlink(missing_ok=True)
    out, _ = proc.communicate()
    result["job_exit_code"] = proc.returncode
    row = parse_log(log)
    row.update(experiment="failure", job="stats", set="full", tag="failure", run=0, combiner=1)
    record(row)
    result["job"] = row
    result["detected_after_kill_s"] = round(t_detect - t_kill, 1) if t_detect else None
    result["rereplicated_after_kill_s"] = round(t_healed - t_kill, 1) if t_healed else None
    text = log.read_text(errors="replace")
    result["task_attempt_failures"] = sorted(set(re.findall(r"Task Id : (attempt_\S+), Status : (\w+)", text)))
    result["nodes_after"] = hdfs("yarn", "node", "-list", "-all", check=False).strip().splitlines()[-4:]
    chk = sh([JOB, "check", "full"], check=False)
    result["result_correct"] = chk.returncode == 0
    result["check_output"] = chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else chk.stderr.strip()
    print(f"  job exit {proc.returncode}, {row['elapsed_s']} s; launched maps {row['launched_maps']}, "
          f"failed {row['failed_maps']}, killed {row['killed_maps']}")
    print("  " + result["check_output"])

    # Bring the worker back and look once more: its old replicas are now surplus.
    sh([CLUSTER, "start", f"dn{victim}"])
    sh([CLUSTER, "start", f"nm{victim}"])
    time.sleep(45)
    result["fsck_after_rejoin"] = fsck(path, "fsck_failure_after_rejoin.txt")
    print(f"  worker restarted: {result['fsck_after_rejoin']['datanodes']} DataNodes, "
          f"over-replicated {result['fsck_after_rejoin']['over_replicated']}, "
          f"status {result['fsck_after_rejoin']['status']}")
    (OUT / "failure.json").write_text(json.dumps(result, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    choices = ["hdfs", "compare", "combiner", "failure"]
    ap.add_argument("--only", nargs="+", choices=choices, default=choices)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    live = hdfs("hdfs", "dfsadmin", "-report", "-live", check=False).count("\nName:")
    if live != 3:
        sys.exit("the Hadoop cluster is not up with 3 DataNodes: scripts/hadoop/cluster.sh up")
    for b in ("mapper_stats", "reducer_stats", "mapper_normalize"):
        if not (REPO / "build" / "hadoop" / b).exists():
            sys.exit("build/hadoop/* missing: run `make`")
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
        "hadoop": hdfs("hadoop", "version").splitlines()[0],
        "cluster": "Docker: 1 NameNode, 1 ResourceManager, 3 DataNodes, 3 NodeManagers "
                   "(2 task containers of 768 MB each), replication 2, block size 128 MiB",
        "config": "scripts/hadoop/config",
    }
    (OUT / "env.json").write_text(json.dumps(env, indent=2))
    if set(args.only) & {"compare", "combiner"} and (OUT / "jobs.csv").exists() and len(args.only) == 4:
        (OUT / "jobs.csv").unlink()
    t0 = time.perf_counter()
    if "hdfs" in args.only:
        exp_hdfs()
    if "compare" in args.only:
        exp_compare(args.runs)
    if "combiner" in args.only:
        exp_combiner(args.runs)
    if "failure" in args.only:
        exp_failure()
    print(f"done in {time.perf_counter() - t0:.0f} s; results in {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
