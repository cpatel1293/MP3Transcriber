"""
/api/transcribe — Proxies the request to OpenAI Whisper API.
For files > 4.5MB, the frontend calls OpenAI directly using the key from /api/get_key.
For smaller files, this endpoint handles it.
"""

import os
import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Return the OpenAI API key for frontend direct upload."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            self._respond(500, {"error": "OPENAI_API_KEY not set."})
            return
        self._respond(200, {"key": api_key})

    def _respond(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._respond(200, {"ok": True})

    def log_message(self, format, *args):
        pass
