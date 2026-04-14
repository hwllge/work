import os
import sys

import cv2
import mediapipe as mp

from gesture_detector import _FingerState, _classify_rps, _get_finger_state, _to_game_gesture


def _ensure_gui_display() -> bool:
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        return True
    if os.path.exists('/tmp/.X11-unix/X0'):
        os.environ['DISPLAY'] = ':0'
        return True
    return False


def _state_map(state: _FingerState):
    return {
        'THUMB': state.thumb,
        'INDEX': state.index,
        'MIDDLE': state.middle,
        'RING': state.ring,
        'PINKY': state.pinky,
    }


def main():
    if not _ensure_gui_display():
        print('[ERROR] No GUI display detected.')
        print('Try: export DISPLAY=:0  or run from desktop terminal.')
        sys.exit(1)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[ERROR] Camera open failed.')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    prev_states = {
        'Left': _FingerState(False, False, False, False, False),
        'Right': _FingerState(False, False, False, False, False),
    }

    # MCP / PIP / DIP / TIP indices for each finger, used for highlight segments.
    finger_joints = {
        'THUMB': (1, 2, 3, 4),
        'INDEX': (5, 6, 7, 8),
        'MIDDLE': (9, 10, 11, 12),
        'RING': (13, 14, 15, 16),
        'PINKY': (17, 18, 19, 20),
    }

    win = 'RPS Debug Check'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        active_labels = set()

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label
                if label not in prev_states:
                    prev_states[label] = _FingerState(False, False, False, False, False)

                state = _get_finger_state(hand_landmarks, prev_states[label])
                prev_states[label] = state
                active_labels.add(label)

                raw = _classify_rps(state)
                gesture = _to_game_gesture(raw)
                gesture_text = gesture if gesture is not None else f'UNKNOWN({raw})'

                lm = hand_landmarks.landmark
                xs = [int(p.x * fw) for p in lm]
                ys = [int(p.y * fh) for p in lm]
                x1 = max(0, min(xs))
                x2 = min(fw - 1, max(xs))
                y1 = max(0, min(ys))
                y2 = min(fh - 1, max(ys))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 255), 2)

                finger_state_map = _state_map(state)
                open_names = [name for name, is_open in finger_state_map.items() if is_open]

                # Base landmarks/connection drawing for quick visual alignment.
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(120, 120, 120), thickness=1, circle_radius=1),
                    mp_draw.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=1),
                )

                # Highlight each finger chain and tip by open/closed 판단.
                for name, (mcp_i, pip_i, dip_i, tip_i) in finger_joints.items():
                    is_open = finger_state_map[name]
                    col = (60, 220, 90) if is_open else (60, 80, 255)

                    mcp = lm[mcp_i]
                    pip = lm[pip_i]
                    dip = lm[dip_i]
                    tip = lm[tip_i]

                    p_mcp = (int(mcp.x * fw), int(mcp.y * fh))
                    p_pip = (int(pip.x * fw), int(pip.y * fh))
                    p_dip = (int(dip.x * fw), int(dip.y * fh))
                    p_tip = (int(tip.x * fw), int(tip.y * fh))

                    cv2.line(frame, p_mcp, p_pip, col, 2)
                    cv2.line(frame, p_pip, p_dip, col, 2)
                    cv2.line(frame, p_dip, p_tip, col, 2)
                    cv2.circle(frame, p_tip, 7, col, -1)

                score = float(handedness.classification[0].score)
                line1 = f'{label} ({score:.2f})  count={state.count}'
                line2 = f'gesture: {gesture_text}'
                line3 = 'open: ' + (', '.join(open_names) if open_names else 'none')

                text_y = max(20, y1 - 34)
                cv2.putText(frame, line1, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, line2, (x1, text_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
                cv2.putText(frame, line3, (x1, text_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)

        # Reset stale per-hand state if a hand disappears.
        for lbl in list(prev_states.keys()):
            if lbl not in active_labels:
                prev_states[lbl] = _FingerState(False, False, False, False, False)

        cv2.putText(frame, 'Q: quit', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
