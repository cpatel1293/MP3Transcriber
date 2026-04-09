"""
/api/refine — Accepts raw transcription text, sends to Claude for cleanup.
Requires ANTHROPIC_API_KEY env var set in Vercel.
"""

import os
import json
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                self._json_response(500, {"error": "ANTHROPIC_API_KEY not configured in Vercel environment variables."})
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            raw_text = data.get("text", "").strip()

            if not raw_text:
                self._json_response(400, {"error": "No text provided."})
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
            self._json_response(200, {"text": refined})

        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _json_response(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
