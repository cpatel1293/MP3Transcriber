"""
/api/refine — Accepts raw transcription text, sends to Claude for cleanup.
"""

import os
import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            import anthropic
        except ImportError:
            self._respond(500, {"error": "anthropic package not installed"})
            return

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self._respond(500, {"error": "ANTHROPIC_API_KEY not set in environment variables."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._respond(400, {"error": "Empty request body."})
                return

            body = self.rfile.read(content_length)
            data = json.loads(body)
            raw_text = data.get("text", "").strip()

            if not raw_text:
                self._respond(400, {"error": "No text provided."})
                return

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system="""You are an expert audio transcription editor. You receive raw, imperfect transcriptions from Whisper speech recognition. Your job:

1. Fix misheard words based on context (homophones, partial words, slurred speech)
2. Add proper punctuation, capitalization, and sentence structure
3. Fix grammar where the speaker clearly intended something different
4. Preserve the speaker's actual meaning, tone, and personality
5. Mark sections you're truly uncertain about with [unclear]
6. If multiple speakers are apparent, label them (Speaker 1, Speaker 2, etc.)
7. Break into logical paragraphs for readability

Return ONLY the corrected transcription. No commentary or explanation.""",
                messages=[{
                    "role": "user",
                    "content": f"Clean up this raw Whisper transcription:\n\n---\n{raw_text}\n---"
                }]
            )

            refined = "".join(b.text for b in message.content if b.type == "text")
            self._respond(200, {"text": refined})

        except Exception as e:
            self._respond(500, {"error": f"Refinement error: {str(e)}"})

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
