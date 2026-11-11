"""零依赖 Web 演示：标准库 http.server + 一个单文件页面。

没有前端构建、没有 npm，是因为这东西要能在医院内网的离线机器上跑起来。
一个需要 300MB node_modules 才能演示的需求洞察工具，本身就过不了"可交付"这关。
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import pipeline

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "web" / "index.html"
_CACHE: dict[str, dict] = {}


def _interviews() -> list[dict]:
    return [
        {
            "id": iv.id,
            "title": iv.title,
            "role": iv.role,
            "department": iv.department,
            "setting": iv.setting,
            "turns": len(iv.transcript),
            "date": iv.date,
        }
        for iv in pipeline.list_interviews()
    ]


def _analyze(interview_id: str) -> dict:
    if interview_id in _CACHE:
        return _CACHE[interview_id]
    path = pipeline.INTERVIEW_DIR / f"{interview_id}.json"
    if not path.exists():
        raise FileNotFoundError(interview_id)
    result = pipeline.analyze(pipeline.load_interview(path)).to_dict()
    _CACHE[interview_id] = result
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "ClinSight/0.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: object, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            if not INDEX.exists():
                self._send(500, b"web/index.html not found", "text/plain; charset=utf-8")
                return
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/interviews":
            self._json(_interviews())
            return
        if parsed.path == "/api/analyze":
            qs = parse_qs(parsed.query)
            interview_id = (qs.get("id") or [""])[0]
            try:
                self._json(_analyze(interview_id))
            except FileNotFoundError:
                self._json({"error": f"unknown interview: {interview_id}"}, 404)
            except Exception as exc:  # 演示环境里把异常回传给前端，好排查
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._json(_analyze(str(payload.get("id", ""))))
        except FileNotFoundError:
            self._json({"error": "unknown interview"}, 404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, *_args) -> None:  # 静默，保持终端干净
        return


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    httpd = None
    for candidate in range(port, port + 12):
        try:
            httpd = ThreadingHTTPServer((host, candidate), Handler)
            port = candidate
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"端口 {port} 起不来，换一个 --port 再试。")

    url = f"http://{host}:{port}/"
    print(f"ClinSight 演示已启动：{url}")
    print("按 Ctrl+C 停止。")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
