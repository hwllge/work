"""
infer_rps_web.py

학습된 MLP(rps_mlp.pt) + MediaPipe HandLandmarker로
라즈베리파이 카메라 영상을 실시간 판정 후 브라우저에 스트리밍.

실행:
    python infer_rps_web.py
    python infer_rps_web.py --model rps_mlp.pt --port 8080 --width 320 --height 240

브라우저 접속:
    http://<라즈베리파이 IP>:8080
"""

import argparse
import threading
import time
import urllib.request
from http import server
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 관절 각도 설정 (train_rps_mlp.py와 동일)
# ─────────────────────────────────────────────
FINGER_CHAINS = [
    [0,  1,  2,  3,  4],
    [0,  5,  6,  7,  8],
    [0,  9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]
WRIST_PAIRS  = [(1, 5), (1, 9), (1, 13), (1, 17)]
NUM_FEATURES = sum(len(c) - 2 for c in FINGER_CHAINS) + len(WRIST_PAIRS)  # 19
LABEL_MAP    = {0: "scissors", 1: "rock", 2: "paper"}
LABEL_KO     = {0: "가위", 1: "바위", 2: "보"}
COLOR_MAP    = {0: (255,  80,  80), 1: (80, 200,  80), 2: (80, 130, 255)}  # BGR

HAND_LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = "hand_landmarker.task"


# ─────────────────────────────────────────────
# MLP 모델 (train_rps_mlp.py와 동일)
# ─────────────────────────────────────────────
class RPSMLP(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, hidden_dims=(64, 32), num_classes=3, dropout=0.3):
        super().__init__()
        layers, dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Feature 추출 (train_rps_mlp.py와 동일)
# ─────────────────────────────────────────────
def _cosine(v1, v2):
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 1.0 if d < 1e-7 else float(np.dot(v1, v2) / d)


def landmarks_to_features(lm: np.ndarray) -> np.ndarray:
    f = []
    for chain in FINGER_CHAINS:
        for i in range(1, len(chain) - 1):
            p0, p1, p2 = lm[chain[i-1]], lm[chain[i]], lm[chain[i+1]]
            f.append(_cosine(p1 - p0, p2 - p1))
    for a, b in WRIST_PAIRS:
        f.append(_cosine(lm[a] - lm[0], lm[b] - lm[0]))
    return np.array(f, dtype=np.float32)


# ─────────────────────────────────────────────
# 모델 / HandLandmarker 초기화
# ─────────────────────────────────────────────
def load_mlp(model_path: str, hidden_dims: list) -> nn.Module:
    model = RPSMLP(hidden_dims=tuple(hidden_dims))
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def get_hand_landmarker(model_path: str = HAND_LANDMARKER_PATH) -> mp_vision.HandLandmarker:
    if not Path(model_path).exists():
        print(f"hand_landmarker.task 다운로드 중 → {model_path}")
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
# 공유 상태
# ─────────────────────────────────────────────
class InferState:
    def __init__(self):
        self.lock       = threading.Lock()
        self.latest_jpg = None
        self.result     = {"label": None, "label_ko": None, "confidence": None}
        self.fps        = 0.0
        self.running    = True


# ─────────────────────────────────────────────
# 카메라 + 추론 루프
# ─────────────────────────────────────────────
def run_infer_loop(state: InferState, args):
    mlp        = load_mlp(args.model, args.hidden)
    landmarker = get_hand_landmarker(args.hand_model)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("카메라 열기 실패. 카메라 인덱스와 권한을 확인하세요.")

    t_prev = time.time()

    while state.running:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        label_id, label_str, label_ko, conf = None, None, None, None

        if result.hand_landmarks:
            lm = np.array(
                [[p.x, p.y, p.z] for p in result.hand_landmarks[0]],
                dtype=np.float32,
            )
            feats = landmarks_to_features(lm)
            with torch.no_grad():
                logits = mlp(torch.tensor(feats).unsqueeze(0))
                probs  = torch.softmax(logits, dim=1)[0].numpy()
            label_id  = int(probs.argmax())
            label_str = LABEL_MAP[label_id]
            label_ko  = LABEL_KO[label_id]
            conf      = float(probs[label_id])
            color     = COLOR_MAP[label_id]

            # 랜드마크 21개 점 표시
            h_px, w_px = frame.shape[:2]
            for pt in result.hand_landmarks[0]:
                cx, cy = int(pt.x * w_px), int(pt.y * h_px)
                cv2.circle(frame, (cx, cy), 4, color, -1)

            # 판정 텍스트
            text = f"{label_str} ({label_ko})  {conf*100:.1f}%"
            cv2.putText(frame, text, (14, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, "No hand detected", (14, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2, cv2.LINE_AA)

        # FPS 계산 및 표시
        now   = time.time()
        fps   = 1.0 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.putText(frame, f"FPS {fps:.1f}", (14, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

        ok2, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        with state.lock:
            if ok2:
                state.latest_jpg = encoded.tobytes()
            state.fps    = fps
            state.result = {
                "label":      label_str,
                "label_ko":   label_ko,
                "confidence": round(conf * 100, 1) if conf is not None else None,
            }

    cap.release()
    landmarker.close()


# ─────────────────────────────────────────────
# HTML 페이지
# ─────────────────────────────────────────────
def build_html():
    return """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>RPS Inference</title>
  <style>
    :root { --bg:#0d1520; --panel:#162033; --accent:#00d4aa; --text:#e8f4ff; --muted:#7a9cc0; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { background:var(--bg); color:var(--text); font-family:'Segoe UI',sans-serif;
           display:flex; flex-direction:column; align-items:center; padding:20px; min-height:100vh; }
    h1 { font-size:1.4rem; color:var(--accent); margin-bottom:14px; letter-spacing:2px; }
    .card { background:var(--panel); border-radius:16px; padding:14px;
            box-shadow:0 12px 32px rgba(0,0,0,.4); max-width:700px; width:100%; }
    img  { width:100%; border-radius:10px; border:1px solid #2a4060; display:block; }
    .result { margin-top:14px; text-align:center; }
    .label  { font-size:2.8rem; font-weight:800; color:var(--accent); line-height:1.2; }
    .conf   { font-size:1.1rem; color:var(--muted); margin-top:4px; }
    .fps    { font-size:0.85rem; color:var(--muted); margin-top:8px; }
    .nohand { font-size:1.2rem; color:#555; }
  </style>
</head>
<body>
  <h1>✊ RPS Inference</h1>
  <div class="card">
    <img src="/stream" alt="camera"/>
    <div class="result">
      <div class="label"  id="label">-</div>
      <div class="conf"   id="conf"></div>
      <div class="fps"    id="fps"></div>
    </div>
  </div>
  <script>
    async function poll() {
      try {
        const r = await fetch('/status');
        const s = await r.json();
        const lbl = document.getElementById('label');
        const conf = document.getElementById('conf');
        if (s.label) {
          lbl.textContent = s.label_ko + '  ' + s.label;
          lbl.style.color = {'scissors':'#ff6060','rock':'#60e060','paper':'#6090ff'}[s.label] || '#00d4aa';
          conf.textContent = '확률 ' + s.confidence + '%';
        } else {
          lbl.textContent  = '손 없음';
          lbl.style.color  = '#555';
          conf.textContent = '';
        }
        document.getElementById('fps').textContent = 'FPS ' + s.fps.toFixed(1);
      } catch(e) {}
    }
    setInterval(poll, 200);
    poll();
  </script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# HTTP 핸들러
# ─────────────────────────────────────────────
def make_handler(state: InferState):
    class Handler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body = build_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == "/status":
                with state.lock:
                    payload = {**state.result, "fps": round(state.fps, 1)}
                import json
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with state.lock:
                            jpg = state.latest_jpg
                        if jpg is None:
                            time.sleep(0.03)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            return

    return Handler


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="RPS real-time inference (web browser)")
    parser.add_argument("--model",      default="rps_mlp.pt",         help="학습된 MLP 가중치 경로")
    parser.add_argument("--hand-model", default=HAND_LANDMARKER_PATH,  help="hand_landmarker.task 경로")
    parser.add_argument("--hidden",     nargs="+", type=int, default=[64, 32],
                        help="모델 은닉층 (학습 시와 동일하게)")
    parser.add_argument("--camera",     type=int,   default=0,         help="카메라 인덱스")
    parser.add_argument("--width",      type=int,   default=320,       help="카메라 해상도 폭")
    parser.add_argument("--height",     type=int,   default=240,       help="카메라 해상도 높이")
    parser.add_argument("--host",       default="0.0.0.0",             help="HTTP 바인드 호스트")
    parser.add_argument("--port",       type=int,   default=8080,      help="HTTP 포트")
    return parser.parse_args()


def main():
    args  = parse_args()
    state = InferState()

    print(f"모델 로드: {args.model}")
    infer_thread = threading.Thread(
        target=run_infer_loop, args=(state, args), daemon=True
    )
    infer_thread.start()

    httpd = server.ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"서버 시작: http://127.0.0.1:{args.port}")
    print(f"같은 네트워크 기기에서: http://<라즈베리파이 IP>:{args.port}")
    print("종료: Ctrl+C")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        httpd.shutdown()


if __name__ == "__main__":
    main()
