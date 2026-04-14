import argparse
import json
import threading
import time
from http import server
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Browser-based RPS dataset collector (MJPEG stream)."
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--out", default="dataset/rps_aug", help="Output directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--size", type=int, default=224, help="Saved image size")
    parser.add_argument("--interval", type=float, default=0.25, help="Auto-capture interval")
    parser.add_argument("--offset", type=int, default=30, help="Hand bbox padding")
    parser.add_argument("--max-hands", type=int, default=1, help="Max hands to detect")
    parser.add_argument("--warmup", type=float, default=1.0, help="Warmup seconds")
    return parser.parse_args()


def make_square_img(img, size):
    h, w = img.shape[:2]
    bg = np.ones((size, size, 3), np.uint8) * 255
    aspect_ratio = h / w

    if aspect_ratio > 1:
        scale = size / h
        resized_w = int(w * scale)
        resized = cv2.resize(img, (resized_w, size))
        pad = (size - resized_w) // 2
        bg[:, pad : pad + resized_w] = resized
    else:
        scale = size / w
        resized_h = int(h * scale)
        resized = cv2.resize(img, (size, resized_h))
        pad = (size - resized_h) // 2
        bg[pad : pad + resized_h, :] = resized

    return bg


def ensure_dirs(base_dir, labels):
    for label in labels:
        (base_dir / label).mkdir(parents=True, exist_ok=True)


def count_images(base_dir, labels):
    counts = {}
    for label in labels:
        label_dir = base_dir / label
        counts[label] = len(
            [p for p in label_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
    return counts


def save_sample(img, base_dir, label, count):
    ts = int(time.time() * 1000)
    filename = f"{label}_{ts}_{count:05d}.jpg"
    path = base_dir / label / filename
    cv2.imwrite(str(path), img)
    return path


class CollectorState:
    def __init__(self, args):
        self.args = args
        self.labels = ["scissors", "rock", "paper"]
        self.out_dir = Path(args.out)
        ensure_dirs(self.out_dir, self.labels)
        self.counts = count_images(self.out_dir, self.labels)

        self.selected_label = "scissors"
        self.auto_mode = False
        self.last_save_time = 0.0
        self.start_time = time.time()

        self.latest_jpeg = None
        self.latest_crop = None
        self.last_frame_time = 0.0
        self.running = True
        self.lock = threading.Lock()
        self.manual_capture_request = False

    def request_capture(self):
        with self.lock:
            self.manual_capture_request = True

    def should_capture_manual(self):
        with self.lock:
            request = self.manual_capture_request
            self.manual_capture_request = False
            return request


def build_html_page():
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RPS Collector</title>
  <style>
    :root {
      --bg: #0e141f;
      --panel: #18263a;
      --accent: #00d1a6;
      --text: #eff6ff;
      --muted: #9fb2cc;
    }
    body { margin: 0; font-family: "Noto Sans KR", sans-serif; background: radial-gradient(circle at top, #1a2f47, var(--bg) 50%); color: var(--text); }
    .wrap { max-width: 980px; margin: 16px auto; padding: 12px; }
    .card { background: linear-gradient(160deg, #1b2b42, var(--panel)); border-radius: 14px; padding: 12px; box-shadow: 0 12px 26px rgba(0,0,0,.28); }
    img { width: 100%; border-radius: 10px; border: 1px solid #365478; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    button { border: 0; border-radius: 9px; padding: 10px 14px; color: #031319; background: #8be9d0; font-weight: 700; cursor: pointer; }
    button.alt { background: #9bb8ff; }
    button.warn { background: #ffd37c; }
    .status { margin-top: 10px; color: var(--muted); line-height: 1.6; }
    code { color: var(--accent); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <img src="/stream" alt="camera stream" />
      <div class="row">
        <button onclick="setLabel('scissors')">scissors</button>
        <button onclick="setLabel('rock')">rock</button>
        <button onclick="setLabel('paper')">paper</button>
        <button class="alt" onclick="toggleAuto()">AUTO ON/OFF</button>
        <button class="warn" onclick="captureOnce()">CAPTURE 1</button>
      </div>
      <div class="status" id="status">loading...</div>
      <div class="status">엔드포인트: <code>/action?cmd=...</code>, <code>/status</code></div>
    </div>
  </div>
  <script>
    async function api(url) {
      const res = await fetch(url, { method: 'GET' });
      return res.json();
    }
    async function setLabel(label) {
      await api('/action?cmd=label&value=' + encodeURIComponent(label));
      refresh();
    }
    async function toggleAuto() {
      await api('/action?cmd=toggle_auto');
      refresh();
    }
    async function captureOnce() {
      await api('/action?cmd=capture');
      refresh();
    }
    async function refresh() {
      try {
        const s = await api('/status');
        document.getElementById('status').textContent =
          `label=${s.label} | auto=${s.auto} | counts: scissors=${s.counts.scissors}, rock=${s.counts.rock}, paper=${s.counts.paper}`;
      } catch (e) {
        document.getElementById('status').textContent = 'status fetch failed';
      }
    }
    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


def run_capture_loop(state, args):
    detector = HandDetector(maxHands=args.max_hands)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Camera open failed. Check camera index and permissions.")

    while state.running:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        hands, _ = detector.findHands(frame, draw=False)
        crop_square = None
        bbox = None

        if hands:
            x, y, w, h = hands[0]["bbox"]
            x1 = max(0, x - args.offset)
            y1 = max(0, y - args.offset)
            x2 = min(frame.shape[1], x + w + args.offset)
            y2 = min(frame.shape[0], y + h + args.offset)

            if x2 > x1 and y2 > y1:
                bbox = (x1, y1, x2, y2)
                crop = frame[y1:y2, x1:x2]
                crop_square = make_square_img(crop, args.size)

        with state.lock:
            state.latest_crop = crop_square

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)

        now = time.time()
        can_capture = (now - state.start_time) >= args.warmup

        manual = state.should_capture_manual()
        with state.lock:
            auto = state.auto_mode
            label = state.selected_label
            interval_ready = (now - state.last_save_time) >= args.interval

        should_auto_save = auto and can_capture and crop_square is not None and interval_ready
        should_manual_save = manual and crop_square is not None

        if should_auto_save or should_manual_save:
            with state.lock:
                label = state.selected_label
                state.counts[label] += 1
                count = state.counts[label]
                state.last_save_time = now
            save_sample(crop_square, state.out_dir, label, count)

        with state.lock:
            info = (
                f"label: {state.selected_label} | mode: {'AUTO' if state.auto_mode else 'MANUAL'} | "
                f"S:{state.counts['scissors']} R:{state.counts['rock']} P:{state.counts['paper']}"
            )

        cv2.putText(
            frame,
            info,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (20, 240, 240),
            2,
            cv2.LINE_AA,
        )

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with state.lock:
                state.latest_jpeg = encoded.tobytes()
                state.last_frame_time = now

    cap.release()


def make_handler(state):
    class Handler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path == "/":
                html = build_html_page().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if parsed.path == "/status":
                with state.lock:
                    payload = {
                        "label": state.selected_label,
                        "auto": state.auto_mode,
                        "counts": state.counts,
                        "last_frame_age": time.time() - state.last_frame_time if state.last_frame_time else None,
                    }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/action":
                query = parse_qs(parsed.query)
                cmd = query.get("cmd", [""])[0]
                value = query.get("value", [""])[0]

                result = {"ok": True}
                with state.lock:
                    if cmd == "label" and value in state.labels:
                        state.selected_label = value
                    elif cmd == "toggle_auto":
                        state.auto_mode = not state.auto_mode
                        state.last_save_time = 0.0
                    elif cmd == "capture":
                        pass
                    else:
                        result = {"ok": False, "error": "invalid cmd"}

                if cmd == "capture":
                    state.request_capture()

                body = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/stream":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                try:
                    while True:
                        with state.lock:
                            jpg = state.latest_jpeg
                        if jpg is None:
                            time.sleep(0.03)
                            continue

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    args = parse_args()
    state = CollectorState(args)

    capture_thread = threading.Thread(target=run_capture_loop, args=(state, args), daemon=True)
    capture_thread.start()

    handler = make_handler(state)
    httpd = server.ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Server started: http://127.0.0.1:{args.port}")
    print("Use browser buttons to select label / toggle auto / capture once.")
    print(f"Saved to: {state.out_dir}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        httpd.shutdown()


if __name__ == "__main__":
    main()