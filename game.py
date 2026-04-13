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
from cvzone.HandTrackingModule import HandDetector

# ─── Hand Detector ───────────────────────────────────────────────────────────
hd = HandDetector(maxHands=1)

# ─── TFLite 모델 로딩 (sample_01 동일 방식) ───────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
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


def _on_mouse(event, x, y, flags, param):
    """터치/마우스 클릭 콜백: 재시작 버튼 영역 탭 감지."""
    if event == cv2.EVENT_LBUTTONDOWN:
        btn = _touch['btn']
        if btn and btn[0] <= x <= btn[2] and btn[1] <= y <= btn[3]:
            _touch['tapped'] = True


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


# ─── 제스처 감지 (sample_01 방식, zone 없음) ─────────────────────────────────
def detect_gesture(frame):
    """현재 프레임에서 (gesture, cx, cy, w) 반환. 손 없으면 (None, None, None, None)."""
    fh, fw = frame.shape[:2]
    hands, _ = hd.findHands(frame, draw=False)
    if not hands:
        return None, None, None, None

    x, y, w, h = hands[0]['bbox']
    x1 = max(0,      x - offset)
    y1 = max(0,      y - offset)
    x2 = min(fw - 1, x + w + offset)
    y2 = min(fh - 1, y + h + offset)
    if x2 <= x1 or y2 <= y1:
        return None, None, None, None

    # 손 중심 좌표
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # crop → make_square → 모델 추론 (sample_01 동일)
    img = frame[y1:y2, x1:x2]
    img = make_square_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    interpreter.set_tensor(input_details[0]['index'], img.astype(input_dtype))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    ans     = int(np.argmax(output_data))
    gesture = ansToText[ans]

    # BB + 아이콘 표시 (sample_01 동일)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[ans], 2)
    cv2.putText(frame, GES_ICON[gesture], (x1, y1 - 7),
                cv2.FONT_HERSHEY_PLAIN, 2, colorList[ans], 2)
    return gesture, cx, cy, w


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
def draw_target(canvas, target, hold_progress=0.0):
    x, y, r   = target['x'], target['y'], target['r']
    ges        = target['gesture']
    idx        = GESTURES.index(ges)
    color      = colorList[idx]
    elapsed    = time.time() - target['start']
    time_ratio = max(0.0, 1.0 - elapsed / TARGET_DURATION)

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
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow('RPS Target Game', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('RPS Target Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('RPS Target Game', _on_mouse)

    g = reset_game()

    while True:
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
            det_ges, hand_cx, hand_cy, hand_w = detect_gesture(frame)
            g['last_det_ges'] = det_ges
            target            = g['target']
            # 라운드가 진행될수록 타겟 지속시간을 지수감소하여 속도 증가
            target_duration = max(MIN_TARGET_DURATION,
                                  TARGET_DURATION * (DECAY_RATE ** g['round_idx']))
            elapsed           = now - target['start']
            hold_progress     = 0.0

            # 위치 판정: 손 중심이 타겟 원 안에 있는지 확인
            in_target = False
            # 손 크기 기준 (너무 작거나 너무 크면 판정 무효화)
            size_ok = False
            if hand_w is not None:
                min_w = int(fw * 0.08)
                max_w = int(fw * 0.75)
                size_ok = (min_w <= hand_w <= max_w)

            if hand_cx is not None:
                dist = ((hand_cx - target['x']) ** 2 + (hand_cy - target['y']) ** 2) ** 0.5
                in_target = size_ok and (dist <= target['r'])
                # 손이 원 안에 있을 때 원을 밝게 표시
                if in_target:
                    cv2.circle(frame, (target['x'], target['y']), target['r'], (255, 255, 255), 2)

            # 디버그: HUD에 hand_w와 허용범위 표시 (짧게)
            if hand_w is not None:
                cv2.putText(frame, f'w:{hand_w}', (10, HUD_H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)
                cv2.putText(frame, f'min:{min_w} max:{max_w}', (10, HUD_H + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200),1)

            if in_target and det_ges == target['gesture']:
                if g['hold_start'] is None:
                    g['hold_start'] = now
                held          = now - g['hold_start']
                hold_progress = min(1.0, held / HOLD_TIME)

                if held >= HOLD_TIME:
                    # 반응 시간 보너스: 빠를수록 최대 TIME_BONUS_MAX 추가
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

        # ── 화면 출력 (sample_01 동일: cv2.imshow + waitKey) ─────────────
        cv2.imshow('RPS Target Game', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        if (key == ord('r') or _touch['tapped']) and g['state'] == 'GAMEOVER':
            _touch['tapped'] = False
            _touch['btn']    = None
            g = reset_game()

    cap.release()
    cv2.destroyAllWindows()
    print(f'Game over! Final score: {g["score"]}')


if __name__ == '__main__':
    main()
