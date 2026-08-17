#!/usr/bin/env python3
"""Offline API contract test for gemini_edit.py using a local HTTP server."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
FAKE_SECRET = "mock-secret"


class Handler(BaseHTTPRequestHandler):
    received_path = None
    received_prompt = None
    received_image = False
    received_auth = False

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        parts = payload["contents"][0]["parts"]
        Handler.received_path = self.path
        Handler.received_prompt = parts[0].get("text")
        Handler.received_image = bool(parts[1].get("inlineData", {}).get("data"))
        Handler.received_auth = self.headers.get("x-goog-api-key") == FAKE_SECRET
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "edited"},
                            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(PNG).decode("ascii")}},
                        ]
                    }
                }
            ]
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        return


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="gemini_edit_test_") as value:
            root = Path(value)
            input_image = root / "input.png"
            output_image = root / "output.png"
            response_json = root / "response.json"
            input_image.write_bytes(PNG)
            env = os.environ.copy()
            env["GEMINI_API_KEY"] = FAKE_SECRET
            env["GEMINI_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
            env.pop("GEMINI_IMAGE_URL", None)
            env.pop("GEMINI_MODEL", None)
            command = [
                sys.executable,
                str(Path(__file__).with_name("gemini_edit.py")),
                "--input-image",
                str(input_image),
                "--output-image",
                str(output_image),
                "--prompt",
                "在墙边添加一个邮箱",
                "--response-json",
                str(response_json),
                "--retries",
                "0",
            ]
            completed = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            combined = completed.stdout + completed.stderr
            if output_image.read_bytes() != PNG or not response_json.is_file():
                raise AssertionError("Gemini mock image/response output mismatch")
            if FAKE_SECRET in combined or FAKE_SECRET in response_json.read_text(encoding="utf-8"):
                raise AssertionError("API key leaked into output")
            expected_path = "/v1beta/models/gemini-3.1-pro-preview:generateContent"
            if (
                Handler.received_path != expected_path
                or Handler.received_prompt != "在墙边添加一个邮箱"
                or not Handler.received_image
                or not Handler.received_auth
            ):
                raise AssertionError(
                    (Handler.received_path, Handler.received_prompt, Handler.received_image, Handler.received_auth)
                )
            print(completed.stdout.strip())
            print("GEMINI_MOCK_READY", Handler.received_path)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
