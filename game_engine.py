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
            'targets': [],
            'hold_starts': [None, None],
            'cleared': [False, False],
            'round_hits': 0,
            'last_round_hits': 0,
            'result_start': None,
            'last_result': None,
            'last_det_ges': None,
        }

    def start_countdown(self, now):
        self.state['state'] = self.COUNTDOWN
        self.state['countdown_start'] = now

    def new_targets(self, fw, fh, now):
        r = self.cfg.target_radius
        targets = []
        for _ in range(2):
            for _attempt in range(30):
                x = random.randint(r + 20, fw - r - 20)
                y = random.randint(self.cfg.hud_h + r + 20, fh - r - 30)
                ok = all(
                    ((x - t['x']) ** 2 + (y - t['y']) ** 2) ** 0.5 >= r * 2 + 60
                    for t in targets
                )
                if ok:
                    break
            targets.append({
                'x': x,
                'y': y,
                'r': r,
                'gesture': random.choice(self.gestures),
                'start': now,
            })
        return targets

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
            self.state['targets'] = self.new_targets(fw, fh, now)
            self.state['hold_starts'] = [None, None]
            self.state['cleared'] = [False, False]
            self.state['round_hits'] = 0
        return elapsed

    def update_playing(self, now, fw, all_hands):
        """all_hands: list of {gesture, cx, cy, w} from detector.detect_all."""
        targets = self.state['targets']
        hold_starts = self.state['hold_starts']
        cleared = self.state['cleared']

        target_duration = self.current_target_duration()
        elapsed = now - targets[0]['start']

        self.state['last_det_ges'] = all_hands[0]['gesture'] if all_hands else None

        min_w = int(fw * 0.04)
        max_w = int(fw * 0.85)

        hold_progresses = [0.0, 0.0]
        in_targets = [False, False]

        for i, target in enumerate(targets):
            if cleared[i]:
                hold_progresses[i] = 1.0
                continue

            best_hand = None
            best_dist = float('inf')
            for hand in all_hands:
                if hand['gesture'] != target['gesture']:
                    continue
                hw = hand.get('w')
                if hw is None or not (min_w <= hw <= max_w):
                    continue
                dist = ((hand['cx'] - target['x']) ** 2 + (hand['cy'] - target['y']) ** 2) ** 0.5
                hit_radius = target['r'] + max(16, int(hw * 0.14))
                if dist <= hit_radius and dist < best_dist:
                    best_dist = dist
                    best_hand = hand

            in_targets[i] = best_hand is not None

            if best_hand is not None:
                if hold_starts[i] is None:
                    hold_starts[i] = now
                held = now - hold_starts[i]
                hold_progresses[i] = min(1.0, held / self.cfg.hold_time)

                if held >= self.cfg.hold_time:
                    time_bonus = int(
                        self.cfg.time_bonus_max * max(0.0, 1.0 - elapsed / target_duration)
                    )
                    raw_score = (self.cfg.base_score + time_bonus) * 2
                    max_raw_per_hit = (self.cfg.base_score + self.cfg.time_bonus_max) * 2
                    max_hits = max(1, self.cfg.total_rounds * len(targets))
                    scale = self.cfg.max_score / float(max_raw_per_hit * max_hits)
                    earned = int(round(raw_score * scale))
                    self.state['score'] = min(self.cfg.max_score, self.state['score'] + earned)
                    cleared[i] = True
                    self.state['round_hits'] += 1
                    hold_starts[i] = None
            else:
                hold_starts[i] = None

        if all(cleared):
            self.state['last_round_hits'] = self.state['round_hits']
            self.state['last_result'] = 'SUCCESS'
            self.state['state'] = self.RESULT
            self.state['result_start'] = now
        elif elapsed >= target_duration:
            self.state['last_round_hits'] = self.state['round_hits']
            self.state['last_result'] = 'FAIL'
            self.state['state'] = self.RESULT
            self.state['result_start'] = now

        first_hand_w = all_hands[0]['w'] if all_hands else None
        return {
            'in_targets': in_targets,
            'hold_progresses': hold_progresses,
            'target_duration': target_duration,
            'hand_w': first_hand_w,
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
        self.state['targets'] = self.new_targets(fw, fh, now)
        self.state['hold_starts'] = [None, None]
        self.state['cleared'] = [False, False]
        self.state['round_hits'] = 0
