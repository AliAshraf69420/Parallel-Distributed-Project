#!/usr/bin/env bash
# Demo of the query service (bonus): normal queries, then the three ways a server can fail
# to answer in time. Serves the results of the MapReduce pipeline for SET (default full;
# run `scripts/hadoop/job.sh all SET` first, or pass --results DIR of any other version).
#
#   scripts/rpc_demo.sh [SET] [extra rpc_server options]
set -u
cd "$(dirname "$0")/.."
SET="${1:-full}"; shift || true
PORT=4010
C="build/rpc_client --port $PORT"
server() { build/rpc_server --set "$SET" --port $PORT "$@" & SRV=$!; sleep 0.5; }
stop() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
say() { printf '\n== %s\n' "$*"; }
run() { printf '$ %s\n' "$*"; "$@"; printf '  (exit status %d)\n' $?; }

say "1. Healthy server: every request is answered"
server "$@"
run $C PING
run $C STATS
run $C CLASS 0
run $C IMAGE "$(sed -n 2p "results/reference/$SET/checksums.csv" | cut -d, -f1)"
run $C IMAGE 999999
stop

say "2. Slow server (2 s per reply), client timeout 500 ms"
server "$@" --delay-ms 2000
run $C --timeout-ms 500 STATS
say "   the same slow server with a 3 s timeout: it was alive all along"
run $C --timeout-ms 3000 STATS
stop

say "3. Hung server (accepts the request, never replies), client timeout 500 ms, 1 retry"
server "$@" --hang
run $C --timeout-ms 500 --retries 1 STATS
stop

say "4. Server process killed: the operating system refuses the connection"
run $C --timeout-ms 500 STATS

cat <<'TXT'

Cases 2 and 3 print the same thing. From the client's side a slow server and a hung (or
crashed, or unreachable) one are indistinguishable: both are "no reply before the timeout".
A longer timeout only moves the line; it never tells the two apart. Case 4 is the lucky
exception: the process died but its machine is still up, so the kernel answers for it with
a connection reset. If the whole machine or the network had failed, there would be no such
answer and the client would see a timeout again.
TXT
