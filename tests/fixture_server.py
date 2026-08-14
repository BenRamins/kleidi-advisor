"""In-process HTTP fixture server for remote-scan tests (F7.S1.T1).

Serves registered path -> bytes content with real Range/206 support so
tests exercise the head-only remote-scan path with zero real network
access. Also supports a no-Range mode (ignores Range headers, always 200s
the full body — simulating a server that doesn't support ranged requests)
and 404s for any unregistered path.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple


def _parse_range(header: str, total: int) -> Tuple[Optional[int], Optional[int]]:
    if not header.startswith("bytes="):
        return None, None
    spec = header[len("bytes=") :]
    start_s, _, end_s = spec.partition("-")
    try:
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else total - 1
    except ValueError:
        return None, None
    end = min(end, total - 1)
    if start < 0 or start > end:
        return None, None
    return start, end


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class FixtureHTTPServer:
    """Construct, mutate `.routes` freely, `.start()`, request, then `.stop()`."""

    def __init__(self, routes: Optional[Dict[str, bytes]] = None, support_range: bool = True):
        self.routes: Dict[str, bytes] = dict(routes or {})
        self.support_range = support_range
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):  # noqa: A002 - stdlib signature
                pass  # keep test output quiet

            def do_GET(self):  # noqa: N802 - stdlib method name
                content = server.routes.get(self.path)
                if content is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                range_header = self.headers.get("Range")
                if server.support_range and range_header:
                    start, end = _parse_range(range_header, len(content))
                    if start is None:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(content)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    chunk = content[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return

                # No-Range mode, or the client sent no Range header: full body, 200.
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        self._httpd = _Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def url_for(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
