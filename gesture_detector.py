from collections import deque
from dataclasses import dataclass, field
import math
from typing import Dict

import cv2
import mediapipe as mp

from config import GameConfig, GestureConfig


class GestureDetector:
    def __init__(self, game_cfg: GameConfig, ges_cfg: GestureConfig):
        self.game_cfg = game_cfg
        self.ges_cfg = ges_cfg
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Lower threshold so the second hand is less likely to be dropped.
        self._min_handedness_score = 0.5
        # Allow smaller hands so distant users are still detected.
        self._min_hand_width_ratio = 0.03
        # Do not aggressively suppress nearby hands.
        self._duplicate_iou_threshold = 0.75
        self._min_label_y = 16

        self._trackers: Dict[str, _HandTracker] = {
            'Left': _HandTracker(),
            'Right': _HandTracker(),
        }
        self._gesture_style = {
            gesture: (self.ges_cfg.color_list[idx], self.ges_cfg.ges_icon[gesture])
            for idx, gesture in enumerate(self.ges_cfg.gestures)
        }

    def detect(self, frame, target=None):
        candidates = self.detect_all(frame)
        if not candidates:
            return None, None, None, None

        selected = _select_candidate(candidates, target)
        return selected['gesture'], selected['cx'], selected['cy'], selected['w']

    def detect_all(self, frame):
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks or not result.multi_handedness:
            for tracker in self._trackers.values():
                tracker.reset()
            return []

        detections = []
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            score = float(handedness.classification[0].score)
            if label not in self._trackers:
                self._trackers[label] = _HandTracker()

            lm = hand_landmarks.landmark
            xs = [int(p.x * fw) for p in lm]
            ys = [int(p.y * fh) for p in lm]
            x1 = max(0, min(xs))
            x2 = min(fw - 1, max(xs))
            y1 = max(0, min(ys))
            y2 = min(fh - 1, max(ys))
            w = max(0, x2 - x1)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append(
                {
                    'label': label,
                    'score': score,
                    'bbox': (x1, y1, x2, y2),
                    'w': w,
                    'cx': cx,
                    'cy': cy,
                    'hand_landmarks': hand_landmarks,
                }
            )

        detections.sort(key=lambda item: item['score'], reverse=True)
        filtered = []
        for det in detections:
            if det['score'] < self._min_handedness_score:
                continue
            if det['w'] < int(fw * self._min_hand_width_ratio):
                continue
            if any(_bbox_iou(det['bbox'], kept['bbox']) > self._duplicate_iou_threshold for kept in filtered):
                continue
            filtered.append(det)

        candidates = []
        active_labels = set()

        for det in filtered:
            label = det['label']
            x1, y1, x2, y2 = det['bbox']
            w = det['w']
            cx = det['cx']
            cy = det['cy']
            hand_landmarks = det['hand_landmarks']
            active_labels.add(label)

            tracker = self._trackers[label]
            state = _get_finger_state(hand_landmarks, tracker.prev_state)
            tracker.prev_state = state
            raw_gesture = _classify_rps(state)
            stable_gesture = tracker.smoother.update(raw_gesture)
            gesture = _to_game_gesture(stable_gesture)

            if gesture in self._gesture_style:
                color, text = self._gesture_style[gesture]
            else:
                color = (180, 180, 180)
                text = '?'

            dx1, dy1, dx2, dy2 = _expand_bbox(det['bbox'], self.game_cfg.offset, fw, fh)
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
            cv2.putText(frame, f'{label} {text}', (dx1, max(self._min_label_y, dy1 - 7)),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            candidates.append(
                {
                    'gesture': gesture,
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'label': label,
                }
            )

        for label, tracker in self._trackers.items():
            if label not in active_labels:
                tracker.reset()

        return candidates


@dataclass
class _FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    @property
    def count(self) -> int:
        return int(self.thumb) + int(self.index) + int(self.middle) + int(self.ring) + int(self.pinky)


@dataclass
class _HandTracker:
    prev_state: _FingerState = field(default_factory=lambda: _FingerState(False, False, False, False, False))
    smoother: '_GestureSmoother' = None

    def __post_init__(self):
        if self.smoother is None:
            self.smoother = _GestureSmoother(maxlen=7)

    def reset(self) -> None:
        self.prev_state = _FingerState(False, False, False, False, False)
        self.smoother.clear()


class _GestureSmoother:
    def __init__(self, maxlen: int = 7) -> None:
        self.history: deque[str] = deque(maxlen=maxlen)

    def update(self, gesture: str) -> str:
        self.history.append(gesture)
        counts: dict[str, int] = {}
        for item in self.history:
            counts[item] = counts.get(item, 0) + 1
        return max(counts, key=counts.get)

    def clear(self) -> None:
        self.history.clear()


def _distance3(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _angle_deg(a, b, c) -> float:
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)

    v1_norm = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    v2_norm = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if v1_norm == 0.0 or v2_norm == 0.0:
        return 0.0

    cos_theta = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (v1_norm * v2_norm)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _is_finger_extended(lm, mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int,
                        prev_open: bool, palm_scale: float) -> bool:
    """Return True only when BOTH conditions hold:
    1. All three joint angles are nearly straight.
    2. Fingertip is far enough from the palm core (not tucked in).
    """
    # ── 1. Angle check (3 joints: MCP-PIP-DIP, PIP-DIP-TIP, MCP-PIP-TIP) ──
    mcp_pip_angle = _angle_deg(lm[mcp_idx], lm[pip_idx], lm[dip_idx])
    pip_dip_angle = _angle_deg(lm[pip_idx], lm[dip_idx], lm[tip_idx])
    mcp_tip_angle = _angle_deg(lm[mcp_idx], lm[pip_idx], lm[tip_idx])
    if prev_open:
        angles_ok = mcp_pip_angle > 125.0 and pip_dip_angle > 125.0 and mcp_tip_angle > 120.0
    else:
        angles_ok = mcp_pip_angle > 135.0 and pip_dip_angle > 135.0 and mcp_tip_angle > 130.0

    # ── 2. Tip-to-palm-core distance check ──────────────────────────────────
    core = _palm_core_xyz(lm)
    tip_to_core = _distance3_xyz(lm[tip_idx], core)
    far_enough = tip_to_core > palm_scale * 1.0

    return angles_ok and far_enough


def _get_finger_state(hand_landmarks, prev_state: _FingerState) -> _FingerState:
    lm = hand_landmarks.landmark

    palm_scale = (_distance3(lm[0], lm[9]) + _distance3(lm[5], lm[17])) / 2.0
    # ── Thumb: same two-condition logic as other fingers ──────────────────
    # Joint indices: CMC(1)-MCP(2)-IP(3)-TIP(4). Use IP angle + MCP angle + tip distance.
    thumb_mcp_ip_angle = _angle_deg(lm[1], lm[2], lm[3])
    thumb_ip_tip_angle = _angle_deg(lm[2], lm[3], lm[4])
    thumb_mcp_tip_angle = _angle_deg(lm[1], lm[2], lm[4])
    core = _palm_core_xyz(lm)
    thumb_tip_to_core = _distance3_xyz(lm[4], core)
    if prev_state.thumb:
        thumb_angles_ok = (thumb_mcp_ip_angle > 125.0
                           and thumb_ip_tip_angle > 125.0
                           and thumb_mcp_tip_angle > 120.0)
        thumb_far_enough = thumb_tip_to_core > palm_scale * 0.82
    else:
        thumb_angles_ok = (thumb_mcp_ip_angle > 133.0
                           and thumb_ip_tip_angle > 133.0
                           and thumb_mcp_tip_angle > 128.0)
        thumb_far_enough = thumb_tip_to_core > palm_scale * 0.90
    thumb_open = thumb_angles_ok and thumb_far_enough

    # ── Four fingers: 3-joint angle + tip-to-core distance ──────────────────
    index_open  = _is_finger_extended(lm,  5,  6,  7,  8, prev_state.index,  palm_scale)
    middle_open = _is_finger_extended(lm,  9, 10, 11, 12, prev_state.middle, palm_scale)
    ring_open   = _is_finger_extended(lm, 13, 14, 15, 16, prev_state.ring,   palm_scale)
    pinky_open  = _is_finger_extended(lm, 17, 18, 19, 20, prev_state.pinky,  palm_scale)

    return _FingerState(
        thumb=thumb_open,
        index=index_open,
        middle=middle_open,
        ring=ring_open,
        pinky=pinky_open,
    )


def _classify_rps(state: _FingerState) -> str:
    if state.count == 0:
        return 'ROCK'
    if state.count == 5:
        return 'PAPER'
    if state.count == 2:
        return 'SCISSOR'
    return 'UNKNOWN'


def _to_game_gesture(gesture: str):
    if gesture == 'SCISSOR':
        return 'SCISSORS'
    if gesture in ('ROCK', 'PAPER'):
        return gesture
    return None


def _select_candidate(candidates, target):
    valid = [c for c in candidates if c['gesture'] is not None]
    pool = valid if valid else candidates

    if target is not None:
        tx, ty = target['x'], target['y']
        return min(pool, key=lambda c: (c['cx'] - tx) ** 2 + (c['cy'] - ty) ** 2)

    return max(pool, key=lambda c: c['w'])


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _is_thumb_inside_fist(lm, palm_scale):
    core = _palm_core_xyz(lm)
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    index_mcp = lm[5]
    middle_mcp = lm[9]
    thumb_ip_angle = _angle_deg(lm[2], lm[3], lm[4])
    thumb_mcp_angle = _angle_deg(lm[1], lm[2], lm[3])
    curled = thumb_ip_angle < 138.0 or thumb_mcp_angle < 132.0

    # If thumb tip and IP are both close to palm core and curled, treat thumb as folded.
    tip_dist = _distance3_xyz(thumb_tip, core)
    ip_dist = _distance3_xyz(thumb_ip, core)
    core_folded = tip_dist < palm_scale * 0.88 and ip_dist < palm_scale * 0.80 and curled

    # In fist pose, thumb tip often stays close to index/middle MCP area and curled.
    tip_to_index_mcp = _distance3(thumb_tip, index_mcp)
    tip_to_middle_mcp = _distance3(thumb_tip, middle_mcp)
    mcp_folded = (
        tip_to_index_mcp < palm_scale * 0.64
        and tip_to_middle_mcp < palm_scale * 0.78
        and curled
    )

    return core_folded or mcp_folded


def _is_finger_folded_near_palm(lm, mcp_idx, pip_idx, dip_idx, tip_idx, palm_scale):
    core = _palm_core_xyz(lm)
    tip = lm[tip_idx]
    pip = lm[pip_idx]
    mcp = lm[mcp_idx]

    tip_to_core = _distance3_xyz(tip, core)
    tip_to_mcp = _distance3(tip, mcp)
    pip_to_mcp = _distance3(pip, mcp)
    if pip_to_mcp == 0.0:
        return True

    # Folded finger tends to keep tip close to palm core, short-reach, and curled joints.
    near_core = tip_to_core < palm_scale * 0.92
    short_reach = tip_to_mcp < pip_to_mcp * 1.38
    pip_angle = _angle_deg(lm[mcp_idx], lm[pip_idx], lm[dip_idx])
    dip_angle = _angle_deg(lm[pip_idx], lm[dip_idx], lm[tip_idx])
    curled = pip_angle < 140.0 or dip_angle < 140.0
    return near_core and short_reach and curled


def _palm_core_xyz(lm):
    core_ids = (0, 5, 9, 13, 17)
    n = len(core_ids)
    cx = sum(lm[i].x for i in core_ids) / n
    cy = sum(lm[i].y for i in core_ids) / n
    cz = sum(lm[i].z for i in core_ids) / n
    return (cx, cy, cz)


def _distance3_xyz(p, xyz):
    x, y, z = xyz
    return math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2 + (p.z - z) ** 2)


def _expand_bbox(bbox, offset, fw, fh):
    x1, y1, x2, y2 = bbox
    nx1 = int(max(0, x1 - offset))
    ny1 = int(max(0, y1 - offset))
    nx2 = int(min(fw - 1, x2 + offset))
    ny2 = int(min(fh - 1, y2 + offset))
    return nx1, ny1, nx2, ny2
