#!/usr/bin/env bash
# Manage the Docker Hadoop cluster. Run from anywhere inside the repository.
#   cluster.sh up            start the cluster and wait until HDFS and YARN are ready
#   cluster.sh status        live DataNodes and NodeManagers
#   cluster.sh exec CMD...   run CMD on the NameNode container, in /work (e.g. hdfs dfs -ls /)
#   cluster.sh sh            shell on the NameNode container
#   cluster.sh stop NODE     stop one container (dn1..dn3, nm1..nm3): the failure experiment
#   cluster.sh start NODE    start it again
#   cluster.sh down          stop and remove everything, including the data in HDFS
set -euo pipefail
cd "$(dirname "$0")"
DC=(docker compose -p sw401hadoop -f docker-compose.yml)
on_nn() { "${DC[@]}" exec -T -w /work namenode "$@"; }
cmd="${1:-}"; shift || true
case "$cmd" in
  up)
    "${DC[@]}" up -d
    echo "waiting for HDFS (3 DataNodes, safe mode off) and YARN (3 NodeManagers) ..."
    for _ in $(seq 1 90); do
      dn=$(on_nn hdfs dfsadmin -report -live 2>/dev/null | grep -c '^Name:' || true)
      nm=$(on_nn yarn node -list 2>/dev/null | grep -c RUNNING || true)
      safe=$(on_nn hdfs dfsadmin -safemode get 2>/dev/null | grep -c OFF || true)
      if [ "$dn" = 3 ] && [ "$nm" = 3 ] && [ "$safe" = 1 ]; then
        echo "ready: $dn DataNodes, $nm NodeManagers"; exit 0
      fi
      sleep 2
    done
    echo "cluster did not become ready (DataNodes=$dn NodeManagers=$nm)" >&2; exit 1
    ;;
  status)
    on_nn hdfs dfsadmin -report 2>/dev/null | grep -E '^(Live|Dead) datanodes|^Name:|^Configured Capacity|^DFS Used:' | head -12
    on_nn yarn node -list -all 2>/dev/null | grep -E 'Total Nodes|RUNNING|LOST|SHUTDOWN'
    ;;
  exec)  on_nn "$@" ;;
  sh)    "${DC[@]}" exec -w /work namenode bash ;;
  stop)  "${DC[@]}" stop -t 0 "$1" ;;
  start) "${DC[@]}" start "$1" ;;
  down)  "${DC[@]}" down -v ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
