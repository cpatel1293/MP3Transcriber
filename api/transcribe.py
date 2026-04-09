"""
/api/transcribe — Accepts audio file, sends to OpenAI Whisper API.
"""

import os
import json
import tempfile
from http.server import BaseHTTPRequestHandler


def parse_multipart(body, content_type):
    """Extract file from multipart form data."""
    try:
        boundary = content_type.split("boundary=")[-1].strip().encode()
        parts = body.split(b"--" + boundary)

        for part in parts:
            if b"filename=" not in part:
                continue

            header_end = part.find(b"\r\n\r\n")
            if header_end == -1:
                continue

            headers_raw = part[:header_end].decode("utf-8", errors="ignore")
            file_data = part[header_end + 4:]

            # Strip trailing \r\n or boundary markers
            for ending in [b"\r\n--", b"\r\n", b"--"]:
                if file_data.endswith(ending):
                    file_data = file_data[: -len(ending)]
                    break

            filename = "upload.mp3"
            for line in headers_raw.split("\r\n"):
                if "filename=" in line:
                    try:
                        fn = line.split('filename="')[1].split('"')[0]
                        if fn:
                            filename = fn
                    except IndexError:
                        pass
                    break

            return file_data, filename

    except Exception as e:
        print(f"Multipart parse error: {e}")

    return None, None


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            import openai
        except ImportError:
            self._respond(500, {"error": "openai package not installed"})
            return

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            self._respond(500, {"error": "OPENAI_API_KEY not set in environment variables."})
            return

        try:
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))

            if content_length == 0:
                self._respond(400, {"error": "Empty request body."})
                return

            body = self.rfile.read(content_length)

            if "multipart/form-data" not in content_type:
                self._respond(400, {"error": "Expected multipart/form-data content type."})
                return

            file_data, filename = parse_multipart(body, content_type)
            if not file_data or len(file_data) == 0:
                self._respond(400, {"error": "No audio file found in request."})
                return

            # Write to temp file
            suffix = ".mp3"
            if "." in filename:
                suffix = "." + filename.rsplit(".", 1)[-1]

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
            tmp.write(file_data)
            tmp.close()

            try:
                client = openai.OpenAI(api_key=api_key)

                with open(tmp.name, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )

                segments = []
                raw_segments = getattr(transcript, "segments", None) or []
                for seg in raw_segments:
                    if isinstance(seg, dict):
                        segments.append({
                            "start": round(seg.get("start", 0), 2),
                            "end": round(seg.get("end", 0), 2),
                            "text": seg.get("text", "").strip(),
                        })
                    else:
                        segments.append({
                            "start": round(getattr(seg, "start", 0), 2),
                            "end": round(getattr(seg, "end", 0), 2),
                            "text": getattr(seg, "text", "").strip(),
                        })

                self._respond(200, {
                    "text": transcript.text,
                    "segments": segments,
                    "language": getattr(transcript, "language", "en"),
                    "duration": getattr(transcript, "duration", 0),
                })

            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        except Exception as e:
            self._respond(500, {"error": f"Transcription error: {str(e)}"})

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
        """Suppress default logging to stderr."""
        pass
