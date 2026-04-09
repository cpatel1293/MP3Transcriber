"""
/api/transcribe — Accepts audio file, sends to OpenAI Whisper API, returns transcription.
Requires OPENAI_API_KEY env var set in Vercel.
"""

import os
import json
import tempfile
from http.server import BaseHTTPRequestHandler
from io import BytesIO


def parse_multipart(body: bytes, content_type: str):
    """Minimal multipart parser to extract uploaded file."""
    boundary = content_type.split("boundary=")[1].encode()
    parts = body.split(b"--" + boundary)
    for part in parts:
        if b"filename=" in part:
            # Extract filename
            header_end = part.find(b"\r\n\r\n")
            headers = part[:header_end].decode("utf-8", errors="ignore")
            file_data = part[header_end + 4:]
            if file_data.endswith(b"\r\n"):
                file_data = file_data[:-2]

            filename = "upload.mp3"
            for line in headers.split("\r\n"):
                if "filename=" in line:
                    fn = line.split('filename="')[1].split('"')[0]
                    if fn:
                        filename = fn
                    break

            # Extract model field
            model = "whisper-1"
            return file_data, filename, model

    return None, None, None


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                self._json_response(500, {"error": "OPENAI_API_KEY not configured in Vercel environment variables."})
                return

            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            file_data, filename, _ = parse_multipart(body, content_type)
            if not file_data:
                self._json_response(400, {"error": "No audio file found in request."})
                return

            # Save to temp file (Whisper API needs a file-like object with a name)
            suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".mp3"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(file_data)
            tmp.close()

            try:
                client = openai.OpenAI(api_key=api_key)

                with open(tmp.name, "rb") as audio_file:
                    # Verbose JSON gives us segments with timestamps
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )

                segments = []
                if hasattr(transcript, "segments") and transcript.segments:
                    for seg in transcript.segments:
                        segments.append({
                            "start": round(seg.get("start", seg.start) if isinstance(seg, dict) else seg.start, 2),
                            "end": round(seg.get("end", seg.end) if isinstance(seg, dict) else seg.end, 2),
                            "text": (seg.get("text", "") if isinstance(seg, dict) else seg.text).strip(),
                        })

                result = {
                    "text": transcript.text,
                    "segments": segments,
                    "language": getattr(transcript, "language", "en"),
                    "duration": getattr(transcript, "duration", 0),
                }

                self._json_response(200, result)

            finally:
                os.unlink(tmp.name)

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
