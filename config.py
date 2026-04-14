from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LanConfig:
    host: str = '0.0.0.0'
    port: int = 5000
    discovery_port: int = 5001
    expected_clients: int = 3
    min_start_players: int = 2
    start_delay_s: float = 3.0


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
    max_score: int = 1500

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

