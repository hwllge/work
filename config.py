import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LanConfig:
    host: str = '0.0.0.0'
    port: int = 5000
    discovery_port: int = 5001
    expected_clients: int = 3


@dataclass(frozen=True)
class GameConfig:
    cam_w: int = 1024
    cam_h: int = 600

    total_rounds: int = 10
    target_duration: float = 3.0
    result_show: float = 0.5
    hold_time: float = 0.3
    target_radius: int = 70
    hud_h: int = 40

    base_score: int = 100
    time_bonus_max: int = 50

    decay_rate: float = 0.95
    min_target_duration: float = 0.6

    img_size: int = 224
    offset: int = 30


@dataclass(frozen=True)
class GestureConfig:
    gestures: List[str] = field(default_factory=lambda: ['SCISSORS', 'ROCK', 'PAPER'])
    ans_to_text: Dict[int, str] = field(default_factory=lambda: {0: 'SCISSORS', 1: 'ROCK', 2: 'PAPER'})
    color_list: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [(80, 200, 255), (80, 255, 80), (255, 80, 80)]
    )
    ges_ko: Dict[str, str] = field(
        default_factory=lambda: {
            'SCISSORS': 'SCISSORS(가위)',
            'ROCK': 'ROCK(바위)',
            'PAPER': 'PAPER(보)',
        }
    )
    ges_icon: Dict[str, str] = field(
        default_factory=lambda: {'SCISSORS': 'V', 'ROCK': 'O', 'PAPER': '='}
    )


def resolve_model_path(script_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(script_path))
    env_path = os.environ.get('RPS_MODEL_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    candidates = [
        os.path.join(script_dir, 'models', 'RPS_MobileNetV2.tflite'),
        os.path.join(script_dir, 'models', 'RPS_MobileNetV2_Augmentation.tflite'),
        os.path.join(script_dir, 'models', 'RPS_MobileNetV2_Augmentation_QAT.tflite'),
        os.path.join(script_dir, 'models', 'RPS_MobileNetV2_Augmentation_PTQ_INT8.tflite'),
        os.path.join(
            script_dir,
            '..',
            'examples',
            '03_CNN_Based_On-Device_AI',
            'RPS_MobileNetV2_Augmentation.tflite',
        ),
        os.path.join(
            script_dir,
            '..',
            'examples',
            '03_CNN_Based_On-Device_AI',
            'RPS_MobileNetV2.tflite',
        ),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        'Could not find RPS TFLite model file. '
        'Set RPS_MODEL_PATH or place a model under miniproject/models/.'
    )
