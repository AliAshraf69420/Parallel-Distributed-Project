#!/usr/bin/env bash
# Test the MapReduce programs without Hadoop, exactly as Hadoop Streaming would run them:
#
#   cat data | mapper | sort | reducer          (Job 1)
#   cat data | mapper_normalize job1-output     (Job 2, map-only)
#
# then build the result files and compare them with the reference, bit for bit.
#   scripts/hadoop/local_test.sh [SET]     (default 100MB; needs data/processed/text/SET.txt)
set -euo pipefail
cd "$(dirname "$0")/../.."
SET="${1:-100MB}"
TXT="data/processed/text/$SET.txt"
OUT="results/runs/hadoop_local/$SET"
[ -f "$TXT" ] || { echo "missing $TXT: run python preprocessing/to_text.py" >&2; exit 1; }
mkdir -p "$OUT"
cat "$TXT" | build/hadoop/mapper_stats 2>/dev/null | LC_ALL=C sort | build/hadoop/reducer_stats 2>/dev/null > "$OUT/job1.tsv"
cat "$TXT" | build/hadoop/mapper_normalize "$OUT/job1.tsv" 2>/dev/null > "$OUT/job2.tsv"
build/hadoop_finalize --set "$SET" --stats "$OUT/job1.tsv" --crcs "$OUT/job2.tsv" --out "$OUT" >/dev/null
"${PYTHON:-python}" tests/check.py --set "$SET" --run "$OUT"
