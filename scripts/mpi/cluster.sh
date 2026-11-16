#!/usr/bin/env bash
# Manage the 4-node Docker MPI cluster. Run from anywhere inside the repository.
# MPI core binding is off: Docker already confines every node to its own 2 physical cores.
#   cluster.sh up            build image, start nodes, compile into build-cluster/
#   cluster.sh run ARGS...   mpirun ARGS... on the cluster, from /work on node1
#                            (uses scripts/mpi/hostfile; ranks fill node1, then node2, ...)
#   cluster.sh spread ARGS.. same, but ranks are placed round-robin, one per node first
#   cluster.sh sh            shell on node1
#   cluster.sh down          stop and remove the nodes
set -euo pipefail
cd "$(dirname "$0")"
export HOST_UID="$(id -u)"
DC=(docker compose -p sw401mpi -f docker-compose.yml)
MPIRUN=(mpirun --hostfile scripts/mpi/hostfile --mca btl_tcp_if_include eth0)
cmd="${1:-}"; shift || true
case "$cmd" in
  up)
    "${DC[@]}" up -d --build
    # The container toolchain may differ from the host's, so build separately.
    "${DC[@]}" exec -T node1 make -C /work -j8 BUILD=build-cluster build-cluster/mpi build-cluster/pingpong
    ;;
  run)    "${DC[@]}" exec -T node1 "${MPIRUN[@]}" --map-by slot --bind-to none "$@" ;;
  spread) "${DC[@]}" exec -T node1 "${MPIRUN[@]}" --map-by node --bind-to none "$@" ;;
  sh)     "${DC[@]}" exec node1 bash ;;
  down)   "${DC[@]}" down ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
