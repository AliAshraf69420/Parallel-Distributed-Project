// Client of the query service, with a timeout on connecting and on waiting for the reply.
//
//   rpc_client [--host 127.0.0.1] [--port 4010] [--timeout-ms 1000] [--retries 0] REQUEST...
//   e.g.  rpc_client STATS      rpc_client CLASS 57      rpc_client IMAGE 3310
//
// Exit status: 0 = OK reply, 1 = ERR reply or usage error, 2 = connection refused/failed,
// 3 = timeout.
//
// On a timeout the client knows only that no reply arrived in time. A slow server, a hung
// server, a crashed machine and a lost packet all look the same from here, so it cannot
// know whether the request was executed. It may retry, which is safe because every request
// is an idempotent read; a non-idempotent operation would need request ids and
// de-duplication on the server.
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "protocol.hpp"

namespace {

using Clock = std::chrono::steady_clock;

int ms_left(Clock::time_point deadline) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    return left > 0 ? int(left) : 0;
}

// Returns 0 = reply received, 2 = connection failed, 3 = timeout.
int call(const std::string& host, int port, int timeout_ms, const std::string& request, std::string& reply,
         std::string& detail) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        detail = "bad host address " + host;
        ::close(fd);
        return 2;
    }
    auto fail = [&](int code, const std::string& why) {
        detail = why;
        ::close(fd);
        return code;
    };
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
        if (errno != EINPROGRESS) return fail(2, std::string("connect: ") + std::strerror(errno));
        pollfd p{fd, POLLOUT, 0};
        if (::poll(&p, 1, ms_left(deadline)) <= 0) return fail(3, "no answer to the connection request");
        int err = 0;
        socklen_t len = sizeof err;
        ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) return fail(2, std::string("connect: ") + std::strerror(err));
    }
    const std::string line = request + "\n";
    if (::write(fd, line.data(), line.size()) != ssize_t(line.size())) return fail(2, "send failed");
    reply.clear();
    char chunk[256];
    for (;;) {
        pollfd p{fd, POLLIN, 0};
        if (::poll(&p, 1, ms_left(deadline)) <= 0) return fail(3, "request sent, no reply");
        const ssize_t n = ::read(fd, chunk, sizeof chunk);
        if (n == 0) return fail(2, "server closed the connection without replying");
        if (n < 0) return fail(2, std::string("read: ") + std::strerror(errno));
        reply.append(chunk, size_t(n));
        const size_t nl = reply.find('\n');
        if (nl != std::string::npos) {
            reply.resize(nl);
            ::close(fd);
            return 0;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1", request;
    int port = rpc::kDefaultPort, timeout_ms = 1000, retries = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--timeout-ms" && i + 1 < argc) timeout_ms = std::atoi(argv[++i]);
        else if (a == "--retries" && i + 1 < argc) retries = std::atoi(argv[++i]);
        else request += (request.empty() ? "" : " ") + a;
    }
    if (request.empty()) {
        std::fprintf(stderr, "usage: rpc_client [--host H] [--port P] [--timeout-ms T] [--retries R] REQUEST...\n");
        return 1;
    }
    int code = 0;
    for (int attempt = 0; attempt <= retries; ++attempt) {
        std::string reply, detail;
        const auto t0 = Clock::now();
        code = call(host, port, timeout_ms, request, reply, detail);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (code == 0) {
            std::printf("%s    [%.2f ms]\n", reply.c_str(), ms);
            return reply.rfind("OK", 0) == 0 ? 0 : 1;
        }
        if (code == 3)
            std::printf("TIMEOUT after %d ms (%s): the server may be slow, hung or crashed, or the "
                        "message was lost; the client cannot tell which%s\n", timeout_ms, detail.c_str(),
                        attempt < retries ? "; retrying (the request is idempotent)" : "");
        else
            std::printf("FAILED (%s)    [%.2f ms]\n", detail.c_str(), ms);
    }
    return code;
}
