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
            'round_idx': 0,        # completed targets
            'rounds_spawned': 0,   # spawned targets
            'targets': [],
            'flashes': [],         # {msg,color,x,y,expire}
            'last_det_ges': [],
            # Compatibility fields kept for existing renderer/src references.
            'hold_starts': [],
            'cleared': [],
            'round_hits': 0,
            'last_round_hits': 0,
            'round_score': 0,
            'last_round_score': 0,
            'perfect_streak': 0,
            'result_start': None,
            'last_result': None,
        }

    def start_countdown(self, now):
        self.state['state'] = self.COUNTDOWN
        self.state['countdown_start'] = now

    def _new_target(self, fw, fh, now, exclude_gestures=None):
        r = self.cfg.target_radius
        available = [g for g in self.gestures if g not in (exclude_gestures or [])]
        gesture = random.choice(available if available else self.gestures)
        return {
            'x': random.randint(r + 20, fw - r - 20),
            'y': random.randint(self.cfg.hud_h + r + 20, fh - r - 30),
            'r': r,
            'gesture': gesture,
            'start': now,
            'hold_start': None,
            'hold_grace_end': None,
        }

    def _spawn_target(self, fw, fh, now, existing_targets):
        exclude_ges = [t['gesture'] for t in existing_targets]
        for _ in range(10):
            t = self._new_target(fw, fh, now, exclude_ges)
            min_dist = self.cfg.target_radius * 3.0
            if all(((t['x'] - e['x']) ** 2 + (t['y'] - e['y']) ** 2) ** 0.5 > min_dist for e in existing_targets):
                return t
        return self._new_target(fw, fh, now, exclude_ges)

    def current_target_duration(self):
        return max(
            self.cfg.min_target_duration,
            self.cfg.target_duration * (self.cfg.decay_rate ** self.state['round_idx']),
        )

    def update_countdown(self, now, fw, fh):
        elapsed = now - self.state['countdown_start']
        if elapsed >= 4.0:
            self.state['state'] = self.PLAYING
            self.state['targets'].clear()
            self.state['rounds_spawned'] = 0
            count = random.randint(1, min(self.cfg.max_targets, self.cfg.total_rounds))
            for _ in range(count):
                self.state['targets'].append(self._spawn_target(fw, fh, now, self.state['targets']))
                self.state['rounds_spawned'] += 1
        return elapsed

    def update_playing(self, now, fw, all_hands):
        self.state['last_det_ges'] = [h['gesture'] for h in all_hands if h.get('gesture')]
        target_duration = self.current_target_duration()

        min_w = int(fw * self.cfg.hand_min_ratio)
        max_w = int(fw * self.cfg.hand_max_ratio)

        targets = self.state['targets']
        hold_progresses = []
        in_targets = []
        targets_to_remove = []
        sounds = []  # sound events to play this frame

        for target in targets:
            elapsed = now - target['start']
            hold_progress = 0.0
            in_target = False

            matched = None
            best_dist = float('inf')
            for det in all_hands:
                hw = det.get('w')
                if hw is None or not (min_w <= hw <= max_w):
                    continue
                if det.get('gesture') != target['gesture']:
                    continue
                dist = ((det['cx'] - target['x']) ** 2 + (det['cy'] - target['y']) ** 2) ** 0.5
                if dist <= target['r'] and dist < best_dist:
                    best_dist = dist
                    matched = det

            if matched is not None:
                in_target = True
                target['hold_grace_end'] = None
                if target['hold_start'] is None:
                    target['hold_start'] = now
                held = now - target['hold_start']
                hold_progress = min(1.0, held / self.cfg.hold_time)

                if held >= self.cfg.hold_time:
                    time_bonus = int(
                        self.cfg.time_bonus_max * max(0.0, 1.0 - elapsed / target_duration + 0.1)
                    )
                    gain = self.cfg.base_score + time_bonus
                    self.state['score'] += gain
                    self.state['round_score'] = gain
                    self.state['round_idx'] += 1
                    self.state['last_round_hits'] = 1
                    self.state['last_result'] = 'SUCCESS'
                    self.state['flashes'].append(
                        {
                            'msg': 'PERFECT!',
                            'color': (0, 255, 128),
                            'x': target['x'],
                            'y': target['y'],
                            'expire': now + self.cfg.result_show,
                        }
                    )
                    sounds.append('perfect')
                    targets_to_remove.append(target)
            else:
                if target['hold_start'] is not None:
                    if target['hold_grace_end'] is None:
                        target['hold_grace_end'] = now + self.cfg.hold_grace
                    elif now >= target['hold_grace_end']:
                        target['hold_start'] = None
                        target['hold_grace_end'] = None

            if target not in targets_to_remove and elapsed >= target_duration:
                self.state['round_idx'] += 1
                self.state['round_score'] = 0
                self.state['last_round_hits'] = 0
                self.state['last_result'] = 'FAIL'
                self.state['flashes'].append(
                    {
                        'msg': 'MISS...',
                        'color': (0, 80, 255),
                        'x': target['x'],
                        'y': target['y'],
                        'expire': now + self.cfg.result_show,
                    }
                )
                sounds.append('miss')
                targets_to_remove.append(target)

            hold_progresses.append(hold_progress)
            in_targets.append(in_target)

        for t in targets_to_remove:
            if t in self.state['targets']:
                self.state['targets'].remove(t)

        if targets_to_remove and self.state['rounds_spawned'] < self.cfg.total_rounds:
            desired = random.randint(1, self.cfg.max_targets)
            while len(self.state['targets']) < desired and self.state['rounds_spawned'] < self.cfg.total_rounds:
                self.state['targets'].append(self._spawn_target(fw, self.cfg.cam_h, now, self.state['targets']))
                self.state['rounds_spawned'] += 1

        self.state['flashes'] = [f for f in self.state['flashes'] if f['expire'] > now]

        if self.state['round_idx'] >= self.cfg.total_rounds and len(self.state['targets']) == 0:
            self.state['state'] = self.GAMEOVER

        # Keep compatibility lists length aligned with current targets.
        self.state['hold_starts'] = [t.get('hold_start') for t in self.state['targets']]
        self.state['cleared'] = [False for _ in self.state['targets']]

        first_hand_w = all_hands[0]['w'] if all_hands else None
        return {
            'in_targets': in_targets,
            'hold_progresses': hold_progresses,
            'target_duration': target_duration,
            'hand_w': first_hand_w,
            'min_w': min_w,
            'max_w': max_w,
            'sounds': sounds,
        }

    def update_result(self, now, fw, fh):
        # RESULT state is not used in main-style gameplay flow.
        return
