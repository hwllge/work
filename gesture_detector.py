import os
import urllib.request
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from config import GameConfig, GestureConfig


_HAND_TASK_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)

FEATURE_FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]
FEATURE_WRIST_PAIRS = [(1, 5), (1, 9), (1, 13), (1, 17)]


def _cosine(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-7:
        return 1.0
    return float(np.dot(v1, v2) / denom)


def _landmarks_to_features(lm):
    feats = []
    for chain in FEATURE_FINGER_CHAINS:
        for i in range(1, len(chain) - 1):
            p0, p1, p2 = lm[chain[i - 1]], lm[chain[i]], lm[chain[i + 1]]
            feats.append(_cosine(p1 - p0, p2 - p1))
    for a, b in FEATURE_WRIST_PAIRS:
        feats.append(_cosine(lm[a] - lm[0], lm[b] - lm[0]))
    return np.array(feats, dtype=np.float32)


class GestureDetector:
    """Main branch new_cls_model pipeline: HandLandmarker + ONNX MLP."""

    def __init__(self, game_cfg: GameConfig, ges_cfg: GestureConfig):
        self.game_cfg = game_cfg
        self.ges_cfg = ges_cfg

        self._landmarker = self._create_hand_landmarker()
        model_path = self._resolve_onnx_path()
        self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name

    def _resolve_onnx_path(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, 'new_cls_model', 'rps_mlp.onnx'),
            os.path.join(script_dir, 'rps_mlp.onnx'),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(
            'rps_mlp.onnx not found. Put it in new_cls_model/ or project root.'
        )

    def _resolve_hand_task_path(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, 'new_cls_model', 'hand_landmarker.task'),
            os.path.join(script_dir, 'hand_landmarker.task'),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p

        dst = candidates[0]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f'[INFO] Download hand_landmarker.task -> {dst}')
        urllib.request.urlretrieve(_HAND_TASK_URL, dst)
        return dst

    def _create_hand_landmarker(self):
        task_path = self._resolve_hand_task_path()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=task_path),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, target=None):
        candidates = self.detect_all(frame)
        if not candidates:
            return None, None, None, None

        selected = candidates[0]
        if target is not None:
            tx, ty = target['x'], target['y']
            selected = min(candidates, key=lambda c: (c['cx'] - tx) ** 2 + (c['cy'] - ty) ** 2)

        return selected['gesture'], selected['cx'], selected['cy'], selected['w']

    def detect_all(self, frame) -> List[dict]:
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detected = self._landmarker.detect(mp_img)
        if not detected.hand_landmarks:
            return []

        out = []
        for hand_lm in detected.hand_landmarks:
            lm = np.array([[p.x, p.y, p.z] for p in hand_lm], dtype=np.float32)
            xs = np.clip((lm[:, 0] * fw).astype(np.int32), 0, fw - 1)
            ys = np.clip((lm[:, 1] * fh).astype(np.int32), 0, fh - 1)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            if x2 <= x1 or y2 <= y1:
                continue

            feats = _landmarks_to_features(lm)
            logits = self._session.run(None, {self._input_name: feats[np.newaxis]})[0][0]
            ans = int(np.argmax(logits))
            gesture = self.ges_cfg.ans_to_text.get(ans)
            if gesture is None:
                continue

            color = self.ges_cfg.color_list[ans]
            icon = self.ges_cfg.ges_icon[gesture]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, icon, (x1, max(16, y1 - 7)), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            out.append(
                {
                    'gesture': gesture,
                    'cx': (x1 + x2) // 2,
                    'cy': (y1 + y2) // 2,
                    'w': max(0, x2 - x1),
                    'label': 'NewClsModel',
                }
            )

        out.sort(key=lambda c: c['w'], reverse=True)
        return out

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
