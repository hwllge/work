import time

import cv2


class GameRenderer:
    def __init__(self, game_cfg, ges_cfg):
        self.cfg = game_cfg
        self.ges_cfg = ges_cfg

    def _draw_button(
        self,
        canvas,
        text,
        rect,
        fill_color,
        border_color=(240, 240, 240),
        text_color=(15, 15, 15),
    ):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill_color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2)
        scale = max(0.7, (x2 - x1) / 360)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        tx = x1 + ((x2 - x1) - tw) // 2
        ty = y1 + ((y2 - y1) + th) // 2
        cv2.putText(
            canvas,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            2,
            cv2.LINE_AA,
        )

    def draw_exit_button(self, canvas):
        fh, fw = canvas.shape[:2]
        bw = max(92, int(fw * 0.10))
        bh = max(34, int(fh * 0.06))
        x1 = fw - bw - 12
        y1 = 8
        rect = (x1, y1, x1 + bw, y1 + bh)
        self._draw_button(canvas, 'EXIT', rect, (70, 70, 180), text_color=(255, 255, 255))
        return rect

    def draw_menu(self, canvas):
        fh, fw = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)

        title = 'RPS TARGET GAME'
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(
            canvas,
            title,
            ((fw - tw) // 2, int(fh * 0.24) + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 220, 255),
            3,
            cv2.LINE_AA,
        )

        subtitle = 'Hit the target with the matching hand gesture'
        (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cv2.putText(
            canvas,
            subtitle,
            ((fw - sw) // 2, int(fh * 0.33)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        bw = int(fw * 0.34)
        bh = int(fh * 0.10)
        bx = (fw - bw) // 2
        run_rect = (bx, int(fh * 0.42), bx + bw, int(fh * 0.42) + bh)
        lan_rect = (bx, int(fh * 0.55), bx + bw, int(fh * 0.55) + bh)
        how_rect = (bx, int(fh * 0.68), bx + bw, int(fh * 0.68) + bh)
        exit_rect = (bx, int(fh * 0.81), bx + bw, int(fh * 0.81) + bh)

        self._draw_button(canvas, 'RUN', run_rect, (0, 185, 95))
        self._draw_button(canvas, 'LAN MULTIPLAYER', lan_rect, (180, 80, 200),
                          text_color=(255, 255, 255))
        self._draw_button(
            canvas,
            'HOW TO PLAY',
            how_rect,
            (55, 135, 255),
            text_color=(255, 255, 255),
        )
        self._draw_button(canvas, 'EXIT', exit_rect, (70, 70, 180), text_color=(255, 255, 255))

        cv2.putText(
            canvas,
            '[Q] Quit',
            (18, fh - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return {'run': run_rect, 'lan': lan_rect, 'howto': how_rect, 'exit': exit_rect}

    def draw_howto(self, canvas):
        fh, fw = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

        title = 'HOW TO PLAY'
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(
            canvas,
            title,
            ((fw - tw) // 2, int(fh * 0.17) + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

        lines = [
            '1) Place your hand inside the target circle.',
            '2) Match the required gesture: SCISSORS / ROCK / PAPER.',
            '3) Hold the correct pose briefly to score.',
            '4) Faster reaction gives extra bonus points.',
            f'5) Survive {self.cfg.total_rounds} rounds for max score!',
        ]
        y = int(fh * 0.30)
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (int(fw * 0.12), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += int(fh * 0.09)

        bw = int(fw * 0.23)
        bh = int(fh * 0.09)
        back_rect = (fw - bw - 40, fh - bh - 28, fw - 40, fh - 28)
        self._draw_button(canvas, 'BACK', back_rect, (90, 90, 90), text_color=(245, 245, 245))
        return {'back': back_rect}

    def draw_target(self, canvas, target, hold_progress=0.0, target_duration=None):
        x, y, r = target['x'], target['y'], target['r']
        ges = target['gesture']
        idx = self.ges_cfg.gestures.index(ges)
        color = self.ges_cfg.color_list[idx]

        duration = target_duration if target_duration is not None else self.cfg.target_duration
        elapsed = time.time() - target['start']
        time_ratio = max(0.0, 1.0 - elapsed / duration)

        overlay = canvas.copy()
        cv2.circle(overlay, (x, y), r, color, -1)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        cv2.circle(canvas, (x, y), r, color, 3)

        if time_ratio > 0:
            angle = int(360 * time_ratio)
            cv2.ellipse(canvas, (x, y), (r + 9, r + 9), -90, 0, angle, color, 3)

        if hold_progress > 0:
            hold_angle = int(360 * hold_progress)
            cv2.ellipse(canvas, (x, y), (r + 18, r + 18), -90, 0, hold_angle, (0, 255, 200), 4)

        sc = r / 70.0
        (tw, th), _ = cv2.getTextSize(self.ges_cfg.ges_icon[ges], cv2.FONT_HERSHEY_SIMPLEX, 1.2 * sc, 2)
        cv2.putText(
            canvas,
            self.ges_cfg.ges_icon[ges],
            (x - tw // 2, y + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2 * sc,
            color,
            2,
            cv2.LINE_AA,
        )

        label = self.ges_cfg.ges_ko[ges]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            canvas,
            label,
            (x - tw // 2, y + r + th + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_hud(self, canvas, score, round_idx, det_ges, perfect_streak=0):
        fh, fw = canvas.shape[:2]
        cv2.rectangle(canvas, (0, 0), (fw, self.cfg.hud_h), (20, 20, 20), -1)
        max_score = getattr(self.cfg, 'max_score', 0)
        score_txt = f'SCORE: {score}' if max_score <= 0 else f'SCORE: {score}/{max_score}'

        cv2.putText(
            canvas,
            score_txt,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 180),
            1,
            cv2.LINE_AA,
        )

        rnd = f'Round {round_idx}/{self.cfg.total_rounds}'
        (tw, _), _ = cv2.getTextSize(rnd, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        cv2.putText(
            canvas,
            rnd,
            (fw - tw - 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        if det_ges:
            if isinstance(det_ges, list):
                valid = [g for g in det_ges if g in self.ges_cfg.gestures]
                if not valid:
                    valid = det_ges
                primary = valid[0] if valid else None
                det_txt = '[ ' + ' | '.join(valid[:2]) + ' ]'
            else:
                primary = det_ges
                det_txt = f'[ {det_ges} ]'

            if primary not in self.ges_cfg.gestures:
                primary = self.ges_cfg.gestures[0]
            idx = self.ges_cfg.gestures.index(primary)
            (tw, _), _ = cv2.getTextSize(det_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(
                canvas,
                det_txt,
                ((fw - tw) // 2, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.ges_cfg.color_list[idx],
                1,
                cv2.LINE_AA,
            )

        if perfect_streak >= 2:
            streak_txt = f'PERFECT STREAK x{perfect_streak}'
            (tw, _), _ = cv2.getTextSize(streak_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.putText(
                canvas,
                streak_txt,
                ((fw - tw) // 2, self.cfg.hud_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (250, 230, 120),
                1,
                cv2.LINE_AA,
            )

    def draw_flashes(self, canvas, flashes, now):
        for f in flashes:
            if f.get('expire', 0) <= now:
                continue
            msg = f.get('msg', '')
            color = f.get('color', (255, 255, 255))
            x = int(f.get('x', 0))
            y = int(f.get('y', 0))
            scale = 0.9
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            tx = x - tw // 2
            ty = y - self.cfg.target_radius - 18
            cv2.putText(canvas, msg, (tx + 2, ty + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 2, cv2.LINE_AA)
            cv2.putText(canvas, msg, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def draw_result(self, canvas, hit_count=0, hit_total=2, round_score=0):
        fh, fw = canvas.shape[:2]
        if hit_count == hit_total:
            msg = 'PERFECT!'
            color = (0, 255, 128)
        elif hit_count is not None and hit_count > 0:
            msg = 'GOOD!'
            color = (80, 220, 255)
        else:
            msg = 'MISS...'
            color = (0, 80, 255)
        if hit_count is not None:
            msg = f'{msg} ({hit_count}/{hit_total})'
        scale = max(1.0, 1.4 * fw / self.cfg.cam_w)
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        tx, ty = (fw - tw) // 2, (fh + th) // 2
        cv2.putText(
            canvas,
            msg,
            (tx + 2, ty + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            msg,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            3,
            cv2.LINE_AA,
        )

        gain_txt = f'+{round_score} pts'
        gain_scale = max(0.65, 0.8 * fw / self.cfg.cam_w)
        (gw, _), _ = cv2.getTextSize(gain_txt, cv2.FONT_HERSHEY_SIMPLEX, gain_scale, 2)
        cv2.putText(
            canvas,
            gain_txt,
            ((fw - gw) // 2, ty + int(44 * fw / self.cfg.cam_w)),
            cv2.FONT_HERSHEY_SIMPLEX,
            gain_scale,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )


    def draw_gameover(self, canvas, score, show_buttons=True):
        fh, fw = canvas.shape[:2]
        sc = fw / self.cfg.cam_w
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        max_score = getattr(
            self.cfg,
            'max_score',
            self.cfg.total_rounds * (self.cfg.base_score + self.cfg.time_bonus_max),
        )
        lines = [
            ('GAME OVER', 1.10 * sc, (0, 210, 255), 2),
            (f'Score: {score} / {max_score}', 0.85 * sc, (255, 255, 255), 2),
        ]
        if show_buttons:
            lines += [
                ('', 0.50 * sc, (150, 150, 150), 1),
                ('[ Q ] Quit', 0.55 * sc, (180, 80, 80), 1),
            ]

        y = int(fh * 0.15)
        for text, scale, color, thick in lines:
            if text:
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                cv2.putText(
                    canvas,
                    text,
                    ((fw - tw) // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    color,
                    thick,
                    cv2.LINE_AA,
                )
            y += int(fh * 0.10)

        if not show_buttons:
            return {}

        btn_sc = 0.75 * sc
        btn_th = 2
        pad = int(18 * sc)
        by = int(fh * 0.72)

        # TAP TO RESTART (left of centre)
        r_txt = 'TAP TO RESTART'
        (rtw, rth), _ = cv2.getTextSize(r_txt, cv2.FONT_HERSHEY_SIMPLEX, btn_sc, btn_th)
        gap = int(fw * 0.03)
        total_w = rtw + pad * 2
        rx = fw // 2 - total_w - gap // 2
        rx2, ry2 = rx + total_w, by + rth + pad * 2
        cv2.rectangle(canvas, (rx, by), (rx2, ry2), (0, 180, 80), -1)
        cv2.rectangle(canvas, (rx, by), (rx2, ry2), (0, 255, 120), 2)
        cv2.putText(canvas, r_txt, (rx + pad, by + rth + pad - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, btn_sc, (10, 10, 10), btn_th, cv2.LINE_AA)

        # MAIN MENU (right of centre)
        m_txt = 'MAIN MENU'
        (mtw, mth), _ = cv2.getTextSize(m_txt, cv2.FONT_HERSHEY_SIMPLEX, btn_sc, btn_th)
        mx = fw // 2 + gap // 2
        mx2, my2 = mx + mtw + pad * 2, by + mth + pad * 2
        cv2.rectangle(canvas, (mx, by), (mx2, my2), (55, 100, 200), -1)
        cv2.rectangle(canvas, (mx, by), (mx2, my2), (100, 160, 255), 2)
        cv2.putText(canvas, m_txt, (mx + pad, by + mth + pad - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, btn_sc, (255, 255, 255), btn_th, cv2.LINE_AA)

        return {'restart': (rx, by, rx2, ry2), 'menu': (mx, by, mx2, my2)}

    def draw_lan_finish_waiting(self, canvas, score):
        """Shown when local player finishes a LAN match before others."""
        fh, fw = canvas.shape[:2]
        sc = fw / self.cfg.cam_w
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        max_score = getattr(
            self.cfg,
            'max_score',
            self.cfg.total_rounds * (self.cfg.base_score + self.cfg.time_bonus_max),
        )

        title = 'YOU FINISHED!'
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * sc, 3)
        cv2.putText(canvas, title, ((fw - tw) // 2, int(fh * 0.28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1 * sc, (0, 220, 255), 3, cv2.LINE_AA)

        score_txt = f'Your score: {score} / {max_score}'
        (sw, _), _ = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.80 * sc, 2)
        cv2.putText(canvas, score_txt, ((fw - sw) // 2, int(fh * 0.44)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80 * sc, (255, 255, 255), 2, cv2.LINE_AA)

        wait_txt = 'Waiting for other players to finish...'
        (ww, _), _ = cv2.getTextSize(wait_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.62 * sc, 1)
        cv2.putText(canvas, wait_txt, ((fw - ww) // 2, int(fh * 0.62)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * sc, (200, 200, 200), 1, cv2.LINE_AA)

        return {}

    def draw_countdown(self, canvas, n):
        fh, fw = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        msg = str(n) if n > 0 else 'GO!'
        color = (0, 200, 255) if n > 0 else (0, 255, 128)
        scale = max(1.5, 2.5 * fw / self.cfg.cam_w)
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
        cv2.putText(
            canvas,
            msg,
            ((fw - tw) // 2, (fh + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            4,
            cv2.LINE_AA,
        )

    # ── LAN screens ──────────────────────────────────────────────────────────

    def draw_lan_room_list(self, canvas, rooms):
        """Show LAN room list first, with Create Room button."""
        fh, fw = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (8, 8, 8), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        title = 'LAN ROOM LIST'
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(canvas, title, ((fw - tw) // 2, int(fh * 0.10) + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

        # List panel
        x1, y1 = int(fw * 0.16), int(fh * 0.20)
        x2, y2 = int(fw * 0.84), int(fh * 0.62)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (28, 28, 28), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 100, 100), 2)

        room_buttons = {}
        if rooms:
            ry = y1 + 42
            for idx, room in enumerate(rooms[:6]):
                row = (
                    f"{room.get('name', 'Room')}  "
                    f"{room.get('host', '')}:{room.get('port', '')}  "
                    f"({room.get('max_players', 4)} players)"
                )
                row_rect = (x1 + 10, ry - 26, x2 - 10, ry + 8)
                cv2.rectangle(canvas, (row_rect[0], row_rect[1]), (row_rect[2], row_rect[3]),
                              (45, 45, 45), -1)
                cv2.putText(canvas, row, (x1 + 18, ry),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
                room_buttons[f'room_{idx}'] = row_rect
                ry += 40
        else:
            msg = 'No rooms discovered yet. Wait a moment or create a room.'
            (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
            cv2.putText(canvas, msg, ((fw - mw) // 2, (y1 + y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (170, 170, 170), 1, cv2.LINE_AA)

        # Buttons
        bw = int(fw * 0.26)
        bh = int(fh * 0.09)
        by = int(fh * 0.72)
        create_rect = ((fw - bw) // 2, by, (fw - bw) // 2 + bw, by + bh)
        back_rect = (28, fh - 70, 160, fh - 24)

        self._draw_button(canvas, 'CREATE ROOM', create_rect, (0, 185, 95), text_color=(255, 255, 255))
        self._draw_button(canvas, 'BACK', back_rect, (70, 70, 70), text_color=(220, 220, 220))

        buttons = {
            'create_room': create_rect,
            'back': back_rect,
        }
        buttons.update(room_buttons)
        return buttons

    def draw_lan_lobby(self, canvas, form):
        """LAN lobby: choose Host or Join, enter player name and server IP.

        form = {
            'mode':  'host' | 'join' | None,
            'focus': 'name' | 'ip',
            'name':  str,
            'ip':    str,
            'error': str | None,
        }
        Returns button dict for InputManager.
        """
        fh, fw = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (8, 8, 8), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Title
        title = 'LAN MULTIPLAYER'
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(canvas, title, ((fw - tw) // 2, int(fh * 0.10) + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

        # Host / Join toggle buttons
        bw = int(fw * 0.22)
        bh = int(fh * 0.09)
        gap = int(fw * 0.04)
        by = int(fh * 0.19)
        hx = fw // 2 - bw - gap // 2
        jx = fw // 2 + gap // 2
        host_rect = (hx, by, hx + bw, by + bh)
        join_rect = (jx, by, jx + bw, by + bh)
        self._draw_button(canvas, 'HOST',
                          host_rect,
                          (0, 185, 95) if form['mode'] == 'host' else (60, 60, 60),
                          text_color=(255, 255, 255))
        self._draw_button(canvas, 'JOIN',
                          join_rect,
                          (55, 135, 255) if form['mode'] == 'join' else (60, 60, 60),
                          text_color=(255, 255, 255))

        # Text fields
        field_defs = [('name', 'Player Name', form['name'])]
        if form['mode'] == 'join':
            field_defs.insert(1, ('ip', 'Server IP', form['ip']))

        fy = int(fh * 0.355)
        field_rects = {}
        picker_buttons = {}
        name_picker_open = bool(form.get('name_picker_open', False))
        name_options = form.get('name_options', [])
        for fid, label, value in field_defs:
            focused = form['focus'] == fid
            lscale = 0.52
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lscale, 1)
            cv2.putText(canvas, label, (int(fw * 0.15), fy),
                        cv2.FONT_HERSHEY_SIMPLEX, lscale,
                        (0, 220, 255) if focused else (180, 180, 180), 1, cv2.LINE_AA)
            fy += lh + 6
            frect = (int(fw * 0.15), fy, int(fw * 0.85), fy + int(fh * 0.075))
            field_rects[fid] = frect
            x1, y1, x2, y2 = frect
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 40), -1)
            border_col = (0, 220, 255) if focused else (120, 120, 120)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 2)
            display = value
            if focused and fid == 'ip':
                display += '|'
            cv2.putText(canvas, display, (x1 + 10, y2 - int((y2 - y1) * 0.25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)

            if fid == 'name':
                arrow = 'v' if name_picker_open else '>'
                cv2.putText(canvas, arrow, (x2 - 22, y2 - int((y2 - y1) * 0.28)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 1, cv2.LINE_AA)
                hint = 'Tap to pick a fun nickname'
                cv2.putText(canvas, hint, (x1 + 10, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1, cv2.LINE_AA)

                if name_picker_open and name_options:
                    top_y = y2 + 6
                    row_h = max(28, int(fh * 0.054))
                    row_gap = 4
                    available_h = max(0, int(fh * 0.80) - top_y)
                    max_visible = max(2, min(len(name_options), available_h // (row_h + row_gap)))
                    for i in range(max_visible):
                        opt = name_options[i]
                        oy1 = top_y + i * (row_h + row_gap)
                        oy2 = oy1 + row_h
                        opt_rect = (x1, oy1, x2, oy2)
                        picker_buttons[f'name_opt_{i}'] = opt_rect
                        sel = (opt == form.get('name'))
                        bg = (56, 96, 56) if sel else (42, 42, 42)
                        border = (0, 220, 255) if sel else (95, 95, 95)
                        cv2.rectangle(canvas, (x1, oy1), (x2, oy2), bg, -1)
                        cv2.rectangle(canvas, (x1, oy1), (x2, oy2), border, 1)
                        cv2.putText(canvas, opt, (x1 + 12, oy2 - int(row_h * 0.28)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
                    fy = top_y + max_visible * (row_h + row_gap) + int(fh * 0.02)
                else:
                    fy = y2 + int(fh * 0.035)
            else:
                fy = y2 + int(fh * 0.035)

        # Confirm button
        confirm_bw = int(fw * 0.28)
        confirm_bh = int(fh * 0.09)
        cx = (fw - confirm_bw) // 2
        cy = max(int(fh * 0.82), min(int(fh * 0.90), fy + int(fh * 0.04)))
        confirm_rect = (cx, cy, cx + confirm_bw, cy + confirm_bh)
        label_str = 'CREATE ROOM' if form['mode'] == 'host' else 'JOIN ROOM'
        enabled = form['mode'] is not None
        self._draw_button(canvas, label_str, confirm_rect,
                          (0, 185, 95) if enabled else (50, 50, 50),
                          text_color=(255, 255, 255) if enabled else (120, 120, 120))

        # Back button
        back_bw = int(fw * 0.14)
        back_bh = int(fh * 0.07)
        back_rect = (30, fh - back_bh - 20, 30 + back_bw, fh - 20)
        self._draw_button(canvas, 'BACK', back_rect, (70, 70, 70), text_color=(220, 220, 220))

        # Error message
        if form.get('error'):
            (ew, _), _ = cv2.getTextSize(form['error'], cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(canvas, form['error'],
                        ((fw - ew) // 2, cy - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1, cv2.LINE_AA)

        buttons = {
            'host': host_rect,
            'join': join_rect,
            'confirm': confirm_rect,
            'back': back_rect,
        }
        buttons.update({f'field_{k}': v for k, v in field_rects.items()})
        buttons.update(picker_buttons)
        return buttons

    def draw_lan_waiting(self, canvas, joined, expected, ready_count, player_name, mode, local_ready):
        """Waiting room shown after room is created or joined."""
        fh, fw = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (8, 8, 8), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        title = 'WAITING FOR PLAYERS...' if mode == 'host' else 'WAITING FOR HOST...'
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.putText(canvas, title, ((fw - tw) // 2, int(fh * 0.22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2, cv2.LINE_AA)

        info = f'Players joined: {joined} / {expected}'
        (iw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
        cv2.putText(canvas, info, ((fw - iw) // 2, int(fh * 0.38)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        ready_info = f'Ready: {ready_count} / {expected}'
        (rw, _), _ = cv2.getTextSize(ready_info, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 1)
        cv2.putText(canvas, ready_info, ((fw - rw) // 2, int(fh * 0.46)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 1, cv2.LINE_AA)

        you = f'You: {player_name}'
        (yw, _), _ = cv2.getTextSize(you, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cv2.putText(canvas, you, ((fw - yw) // 2, int(fh * 0.56)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        note = 'Press READY. Game starts when everyone is ready.'
        (nw, _), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(canvas, note, ((fw - nw) // 2, int(fh * 0.67)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)

        ready_bw = int(fw * 0.25)
        ready_bh = int(fh * 0.09)
        ready_rect = ((fw - ready_bw) // 2, int(fh * 0.76),
                      (fw - ready_bw) // 2 + ready_bw, int(fh * 0.76) + ready_bh)
        if local_ready:
            self._draw_button(canvas, 'READY', ready_rect, (60, 60, 60), text_color=(180, 180, 180))
        else:
            self._draw_button(canvas, 'READY', ready_rect, (0, 185, 95), text_color=(255, 255, 255))

        back_rect = (28, fh - 70, 160, fh - 24)
        self._draw_button(canvas, 'BACK', back_rect, (70, 70, 70), text_color=(220, 220, 220))

        return {'ready': ready_rect, 'back_waiting': back_rect}

    def draw_lan_leaderboard(self, canvas, leaderboard, my_name):
        """Full-screen leaderboard overlay after LAN match ends."""
        def _norm_name(v):
            return (v or '').strip().lower()

        fh, fw = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        title = 'LEADERBOARD'
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.putText(canvas, title, ((fw - tw) // 2, int(fh * 0.13) + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 2, cv2.LINE_AA)

        row_colors = [(255, 215, 0), (192, 192, 192), (205, 127, 50)]
        y = int(fh * 0.28)
        for i, entry in enumerate(leaderboard):
            color = row_colors[i] if i < 3 else (200, 200, 200)
            is_me = _norm_name(entry.get('name', '')) == _norm_name(my_name)
            prefix = f"{i + 1}."
            text = f"{prefix}  {entry.get('name', '')}  —  {entry.get('score', 0)}"
            scale = 0.80 if is_me else 0.70
            thick = 2 if is_me else 1
            if is_me:
                (bw2, bh2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
                bx2 = (fw - bw2) // 2 - 10
                cv2.rectangle(canvas, (bx2, y - bh2 - 4), (bx2 + bw2 + 20, y + 8),
                               (40, 40, 40), -1)
            (tw2, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(canvas, text, ((fw - tw2) // 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
            y += int(fh * 0.10)

        bw_b = int(fw * 0.25)
        bh_b = int(fh * 0.08)
        menu_rect = ((fw - bw_b) // 2, int(fh * 0.86),
                     (fw - bw_b) // 2 + bw_b, int(fh * 0.86) + bh_b)
        self._draw_button(canvas, 'MAIN MENU', menu_rect, (0, 185, 95),
                          text_color=(255, 255, 255))
        return {'menu': menu_rect}
