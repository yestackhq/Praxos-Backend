"""Production entrypoint: serve the API on a DUAL-STACK socket.

Neither uvicorn `--host` value works on Railway:

  --host 0.0.0.0   IPv4 only. Railway's private network is IPv6-only, so
                   praxos-backend.railway.internal reaches nothing and the
                   frontend proxy cannot talk to the API at all.
  --host ::        IPv6 only in this container (the socket comes up with
                   IPV6_V6ONLY set), so Railway's healthcheck — which connects
                   over IPv4 — never reaches the app. Observed directly: the
                   deploy logged "Application startup complete" and then failed
                   five healthchecks without a single request reaching uvicorn.

`socket.create_server(..., dualstack_ipv6=True)` explicitly clears IPV6_V6ONLY,
giving one listener that accepts both families. We bind it here and hand uvicorn
the file descriptor.
"""

from __future__ import annotations

import os
import socket

import uvicorn


def _listener(port: int) -> socket.socket:
    if socket.has_dualstack_ipv6():
        return socket.create_server(
            ("", port), family=socket.AF_INET6, dualstack_ipv6=True, backlog=2048
        )
    # No IPv6 support at all (local dev on an odd host) — IPv4 is fine there.
    return socket.create_server(("0.0.0.0", port), backlog=2048)


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    sock = _listener(port)
    family = "dual-stack IPv4+IPv6" if sock.family == socket.AF_INET6 else "IPv4"
    print(f"praxos api listening on :{port} ({family})", flush=True)
    uvicorn.run("lms_app.main:app", fd=sock.fileno(), log_level="info")


if __name__ == "__main__":
    main()
