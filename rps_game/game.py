# game.py ─ RPS 타겟 게임
# 랜덤 위치에 생성되는 타겟 원에 맞는 가위/바위/보 동작으로 점수 획득
#
# 조작:
#   Q  ─ 종료
#   R  ─ 게임 재시작 (게임오버 화면에서)

# ─── 모듈 로딩 (sample_01 동일 순서) ─────────────────────────────────────────
from ai_edge_litert.interpreter import Interpreter
import numpy as np
import time
import random
import cv2
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from cvzone.HandTrackingModule import HandDetector

# ─── Hand Detector ───────────────────────────────────────────────────────────
hd = HandDetector(maxHands=2)

# ─── TFLite 모델 로딩 (sample_01 동일 방식) ───────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    # 학습한 모델 추가
    os.path.join(_SCRIPT_DIR, '..', 'examples', '03_CNN_Based_On-Device_AI',
                 'RPS_MobileNetV2_Augmentation.tflite'),
    os.path.join(_SCRIPT_DIR, '..', 'examples', '03_CNN_Based_On-Device_AI',
                 'RPS_MobileNetV2.tflite'),
]
MODEL_PATH = next(p for p in _MODEL_CANDIDATES if os.path.isfile(p))

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_dtype    = input_details[0]['dtype']
IMG_SIZE       = 224
offset         = 30



# ─── 비동기 감지 워커 (별도 스레드에서 손 감지 + 제스처 분류) ──────────────────
class AsyncDetector:
    """Background thread: hand detection + gesture classification."""

    def __init__(self):
        self._lock = Lock()
        self._latest_frame = None
        self._has_new = False
        self._results = []
        self._running = True
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._latest_frame = frame.copy()
            self._has_new = True

    def get_results(self):
        with self._lock:
            return list(self._results)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)

    def _loop(self):
        while self._running:
            with self._lock:
                if not self._has_new or self._latest_frame is None:
                    frame = None
                else:
                    frame = self._latest_frame.copy()
                    self._has_new = False
            if frame is None:
                time.sleep(0.005)
                continue
            results = self._detect(frame)
            with self._lock:
                self._results = results

    def _detect(self, frame):
        fh, fw = frame.shape[:2]
        hands, _ = hd.findHands(frame, draw=False)
        if not hands:
            return []
        results = []
        for hand in hands:
            x, y, w, h = hand['bbox']
            x1 = max(0, x - offset)
            y1 = max(0, y - offset)
            x2 = min(fw - 1, x + w + offset)
            y2 = min(fh - 1, y + h + offset)
            if x2 <= x1 or y2 <= y1:
                continue
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            img = frame[y1:y2, x1:x2]
            img = make_square_img(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, 0)
            interpreter.set_tensor(input_details[0]['index'],
                                   img.astype(input_dtype))
            interpreter.invoke()
            output_data = interpreter.get_tensor(
                output_details[0]['index'])[0]
            ans = int(np.argmax(output_data))
            gesture = ansToText[ans]
            results.append({
                'gesture': gesture, 'cx': cx, 'cy': cy, 'w': w,
                'bbox': (x1, y1, x2, y2), 'ans': ans,
            })
        return results

# ─── 카메라 / 화면 해상도 ────────────────────────────────────────────────────
CAM_W, CAM_H = 1024, 600

# ─── 게임 설정 ────────────────────────────────────────────────────────────────
TOTAL_ROUNDS    = 10     # 총 라운드 수
TARGET_DURATION = 3.0    # 타겟 유지 시간 (초)
RESULT_SHOW     = 0.5    # 성공/실패 결과 표시 시간 (초)
HOLD_TIME       = 0.3    # 정답 동작 유지 시간 (초)
TARGET_RADIUS   = 70     # 타겟 원 반지름 (px)
HUD_H           = 40     # 상단 HUD 높이 → 타겟 생성 제외 영역

# ─── 점수 설정 ────────────────────────────────────────────────────────────────
BASE_SCORE     = 100     # 기본 점수
TIME_BONUS_MAX = 50      # 반응속도 최대 보너스

# ─── 난이도/속도 튜닝 (상단에서 조절 가능) ─────────────────────────────────
DECAY_RATE = 0.95         # 라운드당 타겟 지속시간 곱해지는 비율 (지수감소)
MIN_TARGET_DURATION = 0.6 # 타겟 지속시간의 하한 (초)

# ─── 터치 상태 ───────────────────────────────────────────────────────────────
_touch = {'tapped': False, 'btn': None}   # btn: (x1, y1, x2, y2)

# ─── 웹 스트리밍 상태 ─────────────────────────────────────────────────────────
WEB_HOST = '0.0.0.0'
WEB_PORT = 8000
_frame_lock = threading.Lock()
_latest_jpeg = None
_running = True
_restart_requested = False
_quit_requested = False


def _on_mouse(event, x, y, flags, param):
    """터치/마우스 클릭 콜백: 재시작 버튼 영역 탭 감지."""
    if event == cv2.EVENT_LBUTTONDOWN:
        btn = _touch['btn']
        if btn and btn[0] <= x <= btn[2] and btn[1] <= y <= btn[3]:
            _touch['tapped'] = True


def _publish_frame(frame):
    """현재 프레임을 JPEG로 인코딩해 웹 스트리밍 버퍼에 저장."""
    global _latest_jpeg
    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return
    with _frame_lock:
        _latest_jpeg = buf.tobytes()


class _WebHandler(BaseHTTPRequestHandler):
    def _send_html(self):
        html = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RPS Target Game</title>
  <style>
    :root { --bg:#081018; --panel:#101a24; --accent:#41d39a; --text:#e8f0f6; }
    body { margin:0; background:radial-gradient(circle at 20% 20%, #132233, #081018 60%); color:var(--text); font-family:"Noto Sans KR", sans-serif; }
    .wrap { max-width:1100px; margin:18px auto; padding:12px; }
    .bar { display:flex; gap:10px; align-items:center; margin-bottom:10px; }
    button { border:0; border-radius:10px; padding:10px 14px; font-weight:700; cursor:pointer; }
    .r { background:var(--accent); color:#062014; }
    .q { background:#ff6b6b; color:#2b0707; }
    .tip { opacity:.85; font-size:14px; }
    .view { width:100%; border-radius:14px; overflow:hidden; box-shadow:0 15px 45px rgba(0,0,0,.35); border:1px solid #203548; }
    img { width:100%; display:block; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <button class="r" onclick="fetch('/restart', {method:'POST'})">Restart</button>
      <button class="q" onclick="fetch('/quit', {method:'POST'})">Quit</button>
      <div class="tip">게임 화면이 안 보이면 잠시 후 자동 갱신됩니다.</div>
    </div>
    <div class="view"><img src="/stream.mjpg" alt="game stream" /></div>
  </div>
</body>
</html>"""
        body = html.encode('utf-8')
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_not_found(self):
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.write(b'Not Found')

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self._send_html()
            return
        if path != '/stream.mjpg':
            self._send_not_found()
            return

        self.send_response(HTTPStatus.OK)
        self.send_header('Age', '0')
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()

        try:
            while _running:
                with _frame_lock:
                    jpg = _latest_jpeg
                if jpg is None:
                    time.sleep(0.03)
                    continue
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(jpg)}\r\n\r\n'.encode('ascii'))
                self.wfile.write(jpg)
                self.wfile.write(b'\r\n')
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        global _restart_requested, _quit_requested
        path = urlparse(self.path).path
        if path == '/restart':
            _restart_requested = True
        elif path == '/quit':
            _quit_requested = True
        else:
            self._send_not_found()
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def log_message(self, fmt, *args):
        # 요청 로그 스팸 방지
        return


def _start_web_server(host=WEB_HOST, port=WEB_PORT):
    server = ThreadingHTTPServer((host, port), _WebHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server


# ─── 제스처 매핑 ─────────────────────────────────────────────────────────────
ansToText = {0: 'SCISSORS', 1: 'ROCK', 2: 'PAPER'}
colorList  = [(80, 200, 255), (80, 255, 80), (255, 80, 80)]
GES_KO     = {'SCISSORS': 'SCISSORS(가위)', 'ROCK': 'ROCK(바위)', 'PAPER': 'PAPER(보)'}
GES_ICON   = {'SCISSORS': 'V', 'ROCK': 'O', 'PAPER': '='}
GESTURES   = ['SCISSORS', 'ROCK', 'PAPER']


# ─── make_square_img (sample_01 동일) ────────────────────────────────────────
def make_square_img(img):
    ho, wo = img.shape[0], img.shape[1]
    wbg = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    if ho / wo > 1:
        wk  = max(1, int(wo * IMG_SIZE / ho))
        img = cv2.resize(img, (wk, IMG_SIZE))
        d   = (IMG_SIZE - wk) // 2
        wbg[:, d:d + wk] = img
    else:
        hk  = max(1, int(ho * IMG_SIZE / wo))
        img = cv2.resize(img, (IMG_SIZE, hk))
        d   = (IMG_SIZE - hk) // 2
        wbg[d:d + hk, :] = img
    return wbg


# ─── 감지 결과로부터 손 오버레이 그리기 ──────────────────────────────────────
def draw_hand_overlays(frame, det_results):
    """AsyncDetector 결과 리스트를 받아 BB + 아이콘을 프레임에 표시."""
    for det in det_results:
        x1, y1, x2, y2 = det['bbox']
        ans = det['ans']
        gesture = det['gesture']
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[ans], 2)
        cv2.putText(frame, GES_ICON[gesture], (x1, y1 - 7),
                    cv2.FONT_HERSHEY_PLAIN, 2, colorList[ans], 2)


# ─── 타겟 생성 ───────────────────────────────────────────────────────────────
def new_target(fw, fh):
    """화면 내 임의 위치에 타겟 원 생성. UI 영역과 겹치지 않도록 함."""
    r  = TARGET_RADIUS
    # 레이블 공간 고려: 아래 여백 r + 20px 추가
    x  = random.randint(r + 20, fw - r - 20)
    y  = random.randint(HUD_H + r + 20, fh - r - 30)
    return {
        'x':       x,
        'y':       y,
        'r':       r,
        'gesture': random.choice(GESTURES),
        'start':   time.time(),
    }

# ─── 타겟 원 그리기 ───────────────────────────────────────────────────────────
def draw_target(canvas, target, hold_progress=0.0, target_dur=TARGET_DURATION):
    x, y, r   = target['x'], target['y'], target['r']
    ges        = target['gesture']
    idx        = GESTURES.index(ges)
    color      = colorList[idx]
    elapsed    = time.time() - target['start']
    time_ratio = max(0.0, 1.0 - elapsed / target_dur)

    # 반투명 채움
    overlay = canvas.copy()
    cv2.circle(overlay, (x, y), r, color, -1)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # 테두리
    cv2.circle(canvas, (x, y), r, color, 3)

    # 타이머 호 (남은 시간, 시계방향)
    if time_ratio > 0:
        angle = int(360 * time_ratio)
        cv2.ellipse(canvas, (x, y), (r + 9, r + 9), -90, 0, angle, color, 3)

    # 홀드 진행 호 (정답 유지 시 채워짐, 초록)
    if hold_progress > 0:
        hold_angle = int(360 * hold_progress)
        cv2.ellipse(canvas, (x, y), (r + 18, r + 18), -90, 0, hold_angle,
                    (0, 255, 200), 4)

    # 제스처 아이콘 (원 안)
    sc = r / 70.0
    (tw, th), _ = cv2.getTextSize(GES_ICON[ges], cv2.FONT_HERSHEY_SIMPLEX, 1.2 * sc, 2)
    cv2.putText(canvas, GES_ICON[ges], (x - tw // 2, y + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * sc, color, 2, cv2.LINE_AA)

    # 제스처 이름 (원 아래)
    label = GES_KO[ges]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(canvas, label, (x - tw // 2, y + r + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# ─── HUD ─────────────────────────────────────────────────────────────────────
def draw_hud(canvas, score, round_idx, det_ges):
    fh, fw = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (fw, HUD_H), (20, 20, 20), -1)

    # 점수 (왼쪽)
    cv2.putText(canvas, f'SCORE: {score}', (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 180), 1, cv2.LINE_AA)

    # 라운드 (오른쪽)
    rnd = f'Round {round_idx}/{TOTAL_ROUNDS}'
    (tw, _), _ = cv2.getTextSize(rnd, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
    cv2.putText(canvas, rnd, (fw - tw - 10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1, cv2.LINE_AA)

    # 현재 감지 제스처 (가운데)
    if det_ges:
        idx = GESTURES.index(det_ges)
        det_txt = f'[ {det_ges} ]'
        (tw, _), _ = cv2.getTextSize(det_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(canvas, det_txt, ((fw - tw) // 2, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colorList[idx], 1, cv2.LINE_AA)


# ─── 결과 오버레이 ────────────────────────────────────────────────────────────
def draw_result(canvas, success):
    fh, fw = canvas.shape[:2]
    msg   = 'PERFECT!' if success else 'MISS...'
    color = (0, 255, 128) if success else (0, 80, 255)
    scale = max(1.0, 1.4 * fw / CAM_W)
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    tx, ty = (fw - tw) // 2, (fh + th) // 2
    cv2.putText(canvas, msg, (tx + 2, ty + 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(canvas, msg, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 3, cv2.LINE_AA)


# ─── 게임오버 ─────────────────────────────────────────────────────────────────
def draw_gameover(canvas, score):
    fh, fw  = canvas.shape[:2]
    sc      = fw / CAM_W
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    max_score = TOTAL_ROUNDS * (BASE_SCORE + TIME_BONUS_MAX)
    lines = [
        ('GAME OVER',                          1.10 * sc, (0, 210, 255), 2),
        (f'Score: {score} / {max_score}',       0.85 * sc, (255, 255, 255), 2),
        ('',                                    0.50 * sc, (150, 150, 150), 1),
        ('[ Q ] Quit',                          0.55 * sc, (180, 80,  80), 1),
    ]
    y = int(fh * 0.15)
    for text, scale, color, thick in lines:
        if text:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(canvas, text, ((fw - tw) // 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += int(fh * 0.10)

    # ── TAP TO RESTART 버튼 ──
    btn_txt = 'TAP TO RESTART'
    btn_sc  = 0.75 * sc
    btn_th  = 2
    (btw, bth), _ = cv2.getTextSize(btn_txt, cv2.FONT_HERSHEY_SIMPLEX, btn_sc, btn_th)
    pad     = int(18 * sc)
    bx      = (fw - btw) // 2 - pad
    by      = int(fh * 0.72)
    bx2, by2 = bx + btw + pad * 2, by + bth + pad * 2
    # 버튼 배경
    cv2.rectangle(canvas, (bx, by), (bx2, by2), (0, 180, 80), -1)
    cv2.rectangle(canvas, (bx, by), (bx2, by2), (0, 255, 120), 2)
    cv2.putText(canvas, btn_txt,
                (bx + pad, by + bth + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, btn_sc, (10, 10, 10), btn_th, cv2.LINE_AA)
    # 콜백이 참조할 수 있도록 전역 저장
    _touch['btn'] = (bx, by, bx2, by2)


# ─── 카운트다운 ───────────────────────────────────────────────────────────────
def draw_countdown(canvas, n):
    fh, fw  = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    msg   = str(n) if n > 0 else 'GO!'
    color = (0, 200, 255) if n > 0 else (0, 255, 128)
    scale = max(1.5, 2.5 * fw / CAM_W)
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
    cv2.putText(canvas, msg, ((fw - tw) // 2, (fh + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 4, cv2.LINE_AA)


# ─── 게임 상태 초기화 ─────────────────────────────────────────────────────────
def reset_game():
    return {
        'state':           'COUNTDOWN',
        'countdown_start': time.time(),
        'score':           0,
        'round_idx':       0,
        'target':          None,
        'hold_start':      None,
        'result_start':    None,
        'last_result':     None,
        'last_det_ges':    None,
    }


# ─── 메인 루프 (sample_01 방식: cv2.imshow + waitKey) ────────────────────────
def main():
    global _running, _restart_requested, _quit_requested

    server = _start_web_server(WEB_HOST, WEB_PORT)
    print(f'Open browser: http://127.0.0.1:{WEB_PORT}')
    print(f'LAN access   : http://<device-ip>:{WEB_PORT}')

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    g = reset_game()

    try:
        while _running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # 카메라 출력 해상도와 무관하게 항상 CAM_W x CAM_H로 채움
            if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                frame = cv2.resize(frame, (CAM_W, CAM_H))
            fh, fw = frame.shape[:2]
            now = time.time()

            # ── COUNTDOWN ────────────────────────────────────────────────────
            if g['state'] == 'COUNTDOWN':
                elapsed = now - g['countdown_start']
                draw_countdown(frame, max(0, 3 - int(elapsed)))
                if elapsed >= 4.0:
                    g['state']  = 'PLAYING'
                    g['target'] = new_target(fw, fh)

           # ── PLAYING ──────────────────────────────────────────────────────
            elif g['state'] == 'PLAYING':
                det_results = detector.get_results()
                draw_hand_overlays(frame, det_results)

                det_gestures      = [d['gesture'] for d in det_results]
                g['last_det_ges'] = det_gestures
                target            = g['target']
                target_duration = max(MIN_TARGET_DURATION,
                                    TARGET_DURATION * (DECAY_RATE ** g['round_idx']))
                elapsed           = now - target['start']
                hold_progress     = 0.0

                # 모든 손에 대해 타겟 판정 (multi-hand)
                any_match = False
                for det in det_results:
                    hand_cx, hand_cy, hand_w = det['cx'], det['cy'], det['w']
                    min_w = int(fw * 0.08)
                    max_w = int(fw * 0.75)
                    size_ok = (min_w <= hand_w <= max_w)
                    dist = ((hand_cx - target['x']) ** 2
                            + (hand_cy - target['y']) ** 2) ** 0.5
                    in_target = size_ok and (dist <= target['r'])
                    if in_target:
                        cv2.circle(frame, (target['x'], target['y']),
                                target['r'], (255, 255, 255), 2)
                    if in_target and det['gesture'] == target['gesture']:
                        any_match = True

                if any_match:
                    if g['hold_start'] is None:
                        g['hold_start'] = now
                    held          = now - g['hold_start']
                    hold_progress = min(1.0, held / HOLD_TIME)
                    if held >= HOLD_TIME:
                        time_bonus        = int(TIME_BONUS_MAX * max(0.0, 1.0 - elapsed / target_duration))
                        g['score']       += BASE_SCORE + time_bonus
                        g['last_result']  = 'SUCCESS'
                        g['state']        = 'RESULT'
                        g['result_start'] = now
                        g['hold_start']   = None
                else:
                    g['hold_start'] = None

                # 타임아웃
                if g['state'] == 'PLAYING' and elapsed >= target_duration:
                    g['last_result']  = 'FAIL'
                    g['state']        = 'RESULT'
                    g['result_start'] = now

                draw_target(frame, target, hold_progress)
                draw_hud(frame, g['score'], g['round_idx'] + 1, det_ges)

            # ── RESULT ───────────────────────────────────────────────────────
            elif g['state'] == 'RESULT':
                draw_target(frame, g['target'])
                draw_hud(frame, g['score'], g['round_idx'] + 1, g['last_det_ges'])
                draw_result(frame, g['last_result'] == 'SUCCESS')

                if now - g['result_start'] >= RESULT_SHOW:
                    g['round_idx'] += 1
                    if g['round_idx'] >= TOTAL_ROUNDS:
                        g['state'] = 'GAMEOVER'
                    else:
                        g['state']      = 'PLAYING'
                        g['target']     = new_target(fw, fh)
                        g['hold_start'] = None

            # ── GAMEOVER ─────────────────────────────────────────────────────
            elif g['state'] == 'GAMEOVER':
                draw_gameover(frame, g['score'])

            if _restart_requested and g['state'] == 'GAMEOVER':
                _restart_requested = False
                g = reset_game()

            if _quit_requested:
                _running = False

            _publish_frame(frame)
            time.sleep(0.005)
    except KeyboardInterrupt:
        _running = False
    finally:
        cap.release()
        server.shutdown()
        server.server_close()

    print(f'Game over! Final score: {g["score"]}')


if __name__ == '__main__':
    main()
