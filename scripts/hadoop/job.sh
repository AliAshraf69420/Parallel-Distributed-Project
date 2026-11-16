#!/usr/bin/env bash
# Run the project's MapReduce jobs on the Docker Hadoop cluster (scripts/hadoop/cluster.sh up).
#
#   job.sh put SET                    upload data/processed/text/SET.txt to HDFS (/cub/SET.txt)
#   job.sh stats SET [options]        Job 1: channel statistics (mapper + combiner + reducer)
#       --no-combiner                 run without -combiner
#       --reducers N                  number of reduce tasks (default 1)
#       --tag T                       suffix for the log and output names (default "run")
#   job.sh normalize SET [--tag T]    Job 2: map-only normalization, using Job 1's result
#   job.sh check SET                  build stats.json/checksums.csv from the job outputs and
#                                     compare with the reference (needs the conda env active)
#   job.sh all SET                    stats + normalize + check
#
# Each job prints its wall-clock seconds as "elapsed_s=..." and keeps its full client log,
# including all job counters, in results/timings/phase3/logs/. Job outputs are fetched to
# results/runs/hadoop/SET/{job1.tsv, job2.tsv}.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
H=scripts/hadoop/cluster.sh
JAR=/opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.4.1.jar
LOGS=results/timings/phase3/logs
cmd="${1:-}"; set_name="${2:-}"; shift 2 || { sed -n '2,16p' "$0"; exit 2; }
combiner=1; reducers=1; tag=run
while [ $# -gt 0 ]; do
  case "$1" in
    --no-combiner) combiner=0 ;;
    --reducers) reducers="$2"; shift ;;
    --tag) tag="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
OUT=results/runs/hadoop/$set_name
mkdir -p "$LOGS" "$OUT"

run_job() {  # run_job NAME HDFS_OUTPUT ARGS...  -> log, elapsed_s
  local name="$1" out="$2"; shift 2
  local log="$LOGS/$name.log"
  $H exec hdfs dfs -rm -r -f -skipTrash "$out" >/dev/null 2>&1 || true
  local t0; t0=$(date +%s.%N)
  if ! $H exec hadoop jar $JAR -D mapreduce.job.name="$name" "$@" -output "$out" >"$log" 2>&1; then
    echo "job $name FAILED, see $log" >&2; tail -5 "$log" >&2; exit 1
  fi
  echo "elapsed_s=$(echo "$(date +%s.%N) - $t0" | bc)" | tee -a "$log"
}

case "$cmd" in
  put)
    $H exec hdfs dfs -mkdir -p /cub
    $H exec hdfs dfs -put -f "data/processed/text/$set_name.txt" /cub/
    $H exec hdfs dfs -ls /cub | grep "$set_name"
    ;;
  stats)
    args=(-D mapreduce.job.reduces="$reducers"
          -files build/hadoop/mapper_stats,build/hadoop/reducer_stats
          -input "/cub/$set_name.txt" -mapper mapper_stats -reducer reducer_stats)
    [ "$combiner" = 1 ] && args+=(-combiner reducer_stats)
    run_job "stats-$set_name-$tag" "/out/stats-$set_name" "${args[@]}"
    $H exec bash -c "hdfs dfs -cat '/out/stats-$set_name/part-*'" > "$OUT/job1.tsv"
    ;;
  normalize)
    [ -s "$OUT/job1.tsv" ] || { echo "run 'job.sh stats $set_name' first" >&2; exit 1; }
    run_job "normalize-$set_name-$tag" "/out/normalize-$set_name" \
      -D mapreduce.job.reduces=0 \
      -files "build/hadoop/mapper_normalize,$OUT/job1.tsv" \
      -input "/cub/$set_name.txt" -mapper "mapper_normalize job1.tsv"
    $H exec bash -c "hdfs dfs -cat '/out/normalize-$set_name/part-*'" > "$OUT/job2.tsv"
    ;;
  check)
    crcs=(); [ -s "$OUT/job2.tsv" ] && crcs=(--crcs "$OUT/job2.tsv")
    rm -f "$OUT/checksums.csv"
    build/hadoop_finalize --set "$set_name" --stats "$OUT/job1.tsv" "${crcs[@]}" --out "$OUT"
    "${PYTHON:-python}" tests/check.py --set "$set_name" --run "$OUT"
    ;;
  all)
    "$0" stats "$set_name" --tag "$tag"
    "$0" normalize "$set_name" --tag "$tag"
    "$0" check "$set_name"
    ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
