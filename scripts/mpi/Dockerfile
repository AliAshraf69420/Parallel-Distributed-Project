# One node of the MPI test cluster: OpenMPI + sshd running as an unprivileged user whose
# UID matches the host user, so files written to the mounted repository keep your ownership.
FROM archlinux:base

RUN pacman -Syu --noconfirm --needed openmpi openssh gcc make && pacman -Scc --noconfirm

ARG UID=1000
RUN useradd -m -u ${UID} mpi
USER mpi
WORKDIR /home/mpi

# Every node is built from this image, so they all share one key pair: any node can ssh to
# any other without a password. sshd runs unprivileged on port 2222.
RUN mkdir -p .ssh && chmod 700 .ssh \
 && ssh-keygen -q -t ed25519 -N "" -f .ssh/id_ed25519 \
 && cp .ssh/id_ed25519.pub .ssh/authorized_keys \
 && ssh-keygen -q -t ed25519 -N "" -f .ssh/host_ed25519 \
 && printf 'Port 2222\nHostKey /home/mpi/.ssh/host_ed25519\nPidFile /home/mpi/.ssh/sshd.pid\nUsePAM no\nPasswordAuthentication no\nStrictModes no\nAuthorizedKeysFile /home/mpi/.ssh/authorized_keys\n' > .ssh/sshd_config \
 && printf 'Host *\n  Port 2222\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n' > .ssh/config \
 && chmod 600 .ssh/config .ssh/authorized_keys

WORKDIR /work
CMD ["/usr/bin/sshd", "-D", "-e", "-f", "/home/mpi/.ssh/sshd_config"]
