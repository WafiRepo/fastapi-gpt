import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


HOST = "127.0.0.1"
PORT = 8088
API_BASE = os.getenv("INQUIRY_API_BASE", "http://127.0.0.1:8000").rstrip("/")
HTML_FILE = Path(__file__).resolve().parent / "chatbot_inquiry_demo.html"


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if not HTML_FILE.exists():
                self._send_json(404, {"error": "chatbot_inquiry_demo.html not found"})
                return
            content = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if not self.path.startswith("/api/"):
            self._send_json(404, {"error": "Not found"})
            return
        target_path = self.path[len("/api"):]
        target_url = f"{API_BASE}{target_path}"
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            req = urlrequest.Request(
                target_url,
                data=raw_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlrequest.urlopen(req, timeout=30) as resp:
                body = resp.read()
                status = resp.status
                content_type = resp.headers.get("Content-Type", "application/json; charset=utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except HTTPError as e:
            body = e.read() or b'{"error":"upstream error"}'
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except URLError as e:
            self._send_json(502, {"error": "Cannot reach API backend", "detail": str(e.reason)})
        except Exception as e:
            self._send_json(500, {"error": "Proxy internal error", "detail": str(e)})


def main():
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Proxy running: http://{HOST}:{PORT}")
    print(f"Forwarding /api/* to: {API_BASE}")
    server.serve_forever()


if __name__ == "__main__":
    main()

