import random
import time


class GameEngine:
    MENU = 'MENU'
    HOWTO = 'HOWTO'
    LAN_ROOM_LIST = 'LAN_ROOM_LIST'
    LAN_LOBBY = 'LAN_LOBBY'
    LAN_WAITING = 'LAN_WAITING'
    LAN_LEADERBOARD = 'LAN_LEADERBOARD'
    COUNTDOWN = 'COUNTDOWN'
    PLAYING = 'PLAYING'
    RESULT = 'RESULT'
    GAMEOVER = 'GAMEOVER'

    def __init__(self, game_cfg, gestures):
        self.cfg = game_cfg
        self.gestures = gestures
        self.state = {}
        self.reset()

    def reset(self):
        self.state = {
            'state': self.MENU,
            'countdown_start': time.time(),
            'score': 0,
            'round_idx': 0,
            'target': None,
            'hold_start': None,
            'result_start': None,
            'last_result': None,
            'last_det_ges': None,
        }

    def start_countdown(self, now):
        self.state['state'] = self.COUNTDOWN
        self.state['countdown_start'] = now

    def new_target(self, fw, fh, now):
        r = self.cfg.target_radius
        x = random.randint(r + 20, fw - r - 20)
        y = random.randint(self.cfg.hud_h + r + 20, fh - r - 30)
        return {
            'x': x,
            'y': y,
            'r': r,
            'gesture': random.choice(self.gestures),
            'start': now,
        }

    def current_target_duration(self):
        round_idx = self.state['round_idx']
        return max(
            self.cfg.min_target_duration,
            self.cfg.target_duration * (self.cfg.decay_rate ** round_idx),
        )

    def update_countdown(self, now, fw, fh):
        elapsed = now - self.state['countdown_start']
        if elapsed >= 4.0:
            self.state['state'] = self.PLAYING
            self.state['target'] = self.new_target(fw, fh, now)
        return elapsed

    def update_playing(self, now, fw, hand_info):
        det_ges, hand_cx, hand_cy, hand_w = hand_info
        self.state['last_det_ges'] = det_ges

        target = self.state['target']
        elapsed = now - target['start']
        target_duration = self.current_target_duration()
        hold_progress = 0.0

        in_target = False
        size_ok = False
        min_w = None
        max_w = None

        if hand_w is not None:
            min_w = int(fw * 0.06)
            max_w = int(fw * 0.85)
            size_ok = min_w <= hand_w <= max_w

        if hand_cx is not None:
            dist = ((hand_cx - target['x']) ** 2 + (hand_cy - target['y']) ** 2) ** 0.5
            # Add margin so near-center hands are still counted as hit.
            hit_radius = target['r'] + (max(12, int(hand_w * 0.12)) if hand_w is not None else 12)
            in_target = size_ok and (dist <= hit_radius)

        if in_target and det_ges == target['gesture']:
            if self.state['hold_start'] is None:
                self.state['hold_start'] = now
            held = now - self.state['hold_start']
            hold_progress = min(1.0, held / self.cfg.hold_time)

            if held >= self.cfg.hold_time:
                time_bonus = int(
                    self.cfg.time_bonus_max * max(0.0, 1.0 - elapsed / target_duration)
                )
                self.state['score'] += self.cfg.base_score + time_bonus
                self.state['last_result'] = 'SUCCESS'
                self.state['state'] = self.RESULT
                self.state['result_start'] = now
                self.state['hold_start'] = None
        else:
            self.state['hold_start'] = None

        if self.state['state'] == self.PLAYING and elapsed >= target_duration:
            self.state['last_result'] = 'FAIL'
            self.state['state'] = self.RESULT
            self.state['result_start'] = now

        return {
            'in_target': in_target,
            'hold_progress': hold_progress,
            'target_duration': target_duration,
            'hand_w': hand_w,
            'min_w': min_w,
            'max_w': max_w,
        }

    def update_result(self, now, fw, fh):
        if now - self.state['result_start'] < self.cfg.result_show:
            return

        self.state['round_idx'] += 1
        if self.state['round_idx'] >= self.cfg.total_rounds:
            self.state['state'] = self.GAMEOVER
            return

        self.state['state'] = self.PLAYING
        self.state['target'] = self.new_target(fw, fh, now)
        self.state['hold_start'] = None
