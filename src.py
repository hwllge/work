import os
import socket
import sys
import threading
import time
from glob import glob

import cv2

from config import GameConfig, GestureConfig, LanConfig
from game_engine import GameEngine
from gesture_detector import GestureDetector
from input_manager import InputManager, handle_text_input
from lan import LanClient, LanRoomAnnouncer, LanRoomScanner, LanServer, get_local_ip
from renderer import GameRenderer


# ── Display guard ────────────────────────────────────────────────────────────

def _ensure_gui_display() -> bool:
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        return True
    if os.path.exists('/tmp/.X11-unix/X0'):
        os.environ['DISPLAY'] = ':0'
        return True
    return False


def _open_camera(game_cfg):
    """Try multiple camera sources and return the first working capture."""
    candidates = []

    env_idx = os.environ.get('CAMERA_INDEX')
    if env_idx is not None:
        try:
            candidates.append(int(env_idx))
        except ValueError:
            pass

    env_dev = os.environ.get('CAMERA_DEVICE')
    if env_dev:
        candidates.append(env_dev)

    candidates.extend([0, 1, 2, 3])
    candidates.extend(sorted(glob('/dev/video*')))

    seen = set()
    uniq_candidates = []
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        uniq_candidates.append(c)

    for src in uniq_candidates:
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, game_cfg.cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, game_cfg.cam_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, _ = cap.read()
        if ok:
            return cap, src

        cap.release()

    return None, None


# ── Shared LAN state (written by background thread, read by main loop) ───────

class LanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.joined = 0          # clients joined so far (server) / 1 when joined (client)
        self.expected = 1
        self.ready = 0
        self.local_ready = False
        self.started = False     # server sent / client received start signal
        self.leaderboard = None  # list[{'name':..,'score':..}] after match
        self.error = None        # str if something went wrong
        self._ready_event = threading.Event()
        self._score_event = threading.Event()
        self._final_score = 0

    def set_local_ready(self):
        with self.lock:
            if not self.local_ready:
                self.local_ready = True
                self.ready += 1
        self._ready_event.set()

    def set_score(self, score: int):
        self._final_score = score
        self._score_event.set()

    def wait_score(self) -> int:
        self._score_event.wait()
        return self._final_score


LAN_NICKNAMES = [
    'Rockstar Panda',
    'Scissor Ninja',
    'Paper Wizard',
    'Thumbless Titan',
    'Combo Cheetah',
    'Lucky Lizard',
    'Turbo Turtle',
    'Captain Crunch',
]


# ── Background threads ───────────────────────────────────────────────────────

def _server_thread(server: LanServer, lan_st: LanState):
    try:
        def _on_ready_progress(ready_count: int, expected: int):
            with lan_st.lock:
                lan_st.ready = ready_count
                lan_st.joined = expected
                lan_st.expected = expected

        server.wait_all_ready(lan_st._ready_event, on_progress=_on_ready_progress)
        server.send_start()
        with lan_st.lock:
            lan_st.started = True

        score = lan_st.wait_score()
        # Server itself is also a player — prepend its own score
        leaderboard = server.collect_scores()
        leaderboard.append({'name': '_host_', 'score': score})
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        server.broadcast_leaderboard(leaderboard)
        with lan_st.lock:
            lan_st.leaderboard = leaderboard
    except Exception as e:
        with lan_st.lock:
            lan_st.error = str(e)
    finally:
        server.close()


def _client_thread(client: LanClient, lan_st: LanState):
    try:
        ack = client.join()
        with lan_st.lock:
            lan_st.joined = ack.get('joined', 1)
            lan_st.expected = ack.get('expected', 3)
            lan_st.ready = 1 if lan_st.local_ready else 0

        lan_st._ready_event.wait()
        client.send_ready()

        def _on_ready_state(ready_count: int, expected: int):
            with lan_st.lock:
                lan_st.ready = ready_count
                lan_st.expected = expected
                lan_st.joined = expected

        client.wait_start(on_ready_state=_on_ready_state)
        with lan_st.lock:
            lan_st.started = True

        score = lan_st.wait_score()
        lb = client.send_score_and_get_leaderboard(score)
        with lan_st.lock:
            lan_st.leaderboard = lb
    except Exception as e:
        with lan_st.lock:
            lan_st.error = str(e)
    finally:
        client.close()


# ── Game loop ────────────────────────────────────────────────────────────────

def run(game_cfg, ges_cfg, lan_cfg):
    if not _ensure_gui_display():
        print('[ERROR] No GUI display detected.')
        print('Try: export DISPLAY=:0  or run from desktop terminal.')
        sys.exit(1)

    renderer = GameRenderer(game_cfg, ges_cfg)
    engine = GameEngine(game_cfg, ges_cfg.gestures)
    input_mgr = InputManager()

    cap, opened_src = _open_camera(game_cfg)
    if cap is None:
        print('[ERROR] Camera open failed.')
        devices = sorted(glob('/dev/video*'))
        print(f"[INFO] Detected video devices: {devices if devices else 'none'}")
        print('[INFO] You can set CAMERA_INDEX or CAMERA_DEVICE env var.')
        sys.exit(1)

    print(f'[INFO] Camera source: {opened_src}')

    detector = GestureDetector(game_cfg, ges_cfg)

    WIN = 'RPS Target Game'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WIN, input_mgr.on_mouse)

    # LAN lobby form state
    lobby = {
        'mode': None,          # 'host' | 'join'
        'focus': 'name',
        'name': LAN_NICKNAMES[0],
        'name_picker_open': False,
        'name_options': LAN_NICKNAMES,
        'ip': '',
        'rooms': [],
        'error': None,
    }
    lan_st: LanState | None = None  # created when room is confirmed
    _server: LanServer | None = None
    _client: LanClient | None = None
    _announcer: LanRoomAnnouncer | None = None
    scanner = LanRoomScanner(lan_cfg.discovery_port)
    scanner.start()

    key = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        if frame.shape[1] != game_cfg.cam_w or frame.shape[0] != game_cfg.cam_h:
            frame = cv2.resize(frame, (game_cfg.cam_w, game_cfg.cam_h))

        now = time.time()
        fh, fw = frame.shape[:2]
        clicked = input_mgr.consume_click()
        state = engine.state['state']
        buttons = {}

        # ── MENU ─────────────────────────────────────────────────────────
        if state == GameEngine.MENU:
            buttons = renderer.draw_menu(frame)
            if clicked == 'run':
                engine.reset()
                engine.start_countdown(now)
            elif clicked == 'howto':
                engine.state['state'] = GameEngine.HOWTO
            elif clicked == 'lan':
                engine.state['state'] = GameEngine.LAN_ROOM_LIST

        # ── HOW TO PLAY ───────────────────────────────────────────────────
        elif state == GameEngine.HOWTO:
            buttons = renderer.draw_howto(frame)
            if clicked == 'back':
                engine.state['state'] = GameEngine.MENU

        # ── LAN ROOM LIST (first screen) ────────────────────────────────
        elif state == GameEngine.LAN_ROOM_LIST:
            lobby['rooms'] = scanner.get_rooms()
            buttons = renderer.draw_lan_room_list(frame, lobby['rooms'])
            if clicked == 'back':
                engine.state['state'] = GameEngine.MENU
            elif clicked == 'create_room':
                lobby['mode'] = 'host'
                lobby['focus'] = 'name'
                lobby['error'] = None
                engine.state['state'] = GameEngine.LAN_LOBBY
            elif clicked == 'join_manual':
                lobby['mode'] = 'join'
                lobby['focus'] = 'ip'
                lobby['error'] = None
                engine.state['state'] = GameEngine.LAN_LOBBY
            elif clicked and clicked.startswith('room_'):
                try:
                    idx = int(clicked.split('_')[1])
                    room = lobby['rooms'][idx]
                    lobby['mode'] = 'join'
                    lobby['ip'] = room.get('host', '')
                    lobby['focus'] = 'name'
                    lobby['error'] = None
                    engine.state['state'] = GameEngine.LAN_LOBBY
                except Exception:
                    pass

        # ── LAN LOBBY (form) ──────────────────────────────────────────────
        elif state == GameEngine.LAN_LOBBY:
            buttons = renderer.draw_lan_lobby(frame, lobby)

            # Focus switch via click on field labels
            for fname in ('ip',):
                if clicked == f'field_{fname}':
                    lobby['focus'] = fname

            if clicked == 'field_name':
                lobby['name_picker_open'] = not lobby.get('name_picker_open', False)
                lobby['focus'] = 'name'

            if clicked and clicked.startswith('name_opt_'):
                try:
                    idx = int(clicked.split('_')[-1])
                    options = lobby.get('name_options', [])
                    if 0 <= idx < len(options):
                        lobby['name'] = options[idx]
                        lobby['name_picker_open'] = False
                except Exception:
                    pass

            if clicked == 'host':
                lobby['mode'] = 'host'
                lobby['name_picker_open'] = False
            elif clicked == 'join':
                lobby['mode'] = 'join'
                lobby['name_picker_open'] = False
            elif clicked == 'back':
                engine.state['state'] = GameEngine.LAN_ROOM_LIST
                lobby['error'] = None
                lobby['name_picker_open'] = False

            elif clicked == 'confirm' and lobby['mode']:
                # Validate
                if not lobby['name'].strip():
                    lobby['error'] = 'Player name is required.'
                elif lobby['mode'] == 'join' and not lobby['ip'].strip():
                    lobby['error'] = 'Server IP is required.'
                else:
                    lobby['error'] = None
                    lan_st = LanState()
                    lan_st.expected = 1

                    if lobby['mode'] == 'host':
                        _server = LanServer('0.0.0.0', lan_cfg.port, lan_cfg.expected_clients)
                        _server.start()
                        lan_st.joined = 1
                        _announcer = LanRoomAnnouncer(
                            room_name=f"{lobby['name']}'s room",
                            host=get_local_ip(),
                            port=lan_cfg.port,
                            discovery_port=lan_cfg.discovery_port,
                            max_players=lan_cfg.expected_clients + 1,
                        )
                        _announcer.start()
                        t = threading.Thread(target=_server_thread,
                                             args=(_server, lan_st), daemon=True)
                        t.start()
                    else:
                        _client = LanClient(lobby['ip'].strip(), lan_cfg.port, lobby['name'].strip())
                        try:
                            _client.connect()
                            t = threading.Thread(target=_client_thread,
                                                 args=(_client, lan_st), daemon=True)
                            t.start()
                        except Exception as e:
                            lobby['error'] = f'Cannot connect: {e}'
                            lan_st = None
                            _client = None

                    if lan_st is not None:
                        engine.state['state'] = GameEngine.LAN_WAITING

            # Key input for focused field
            if state == GameEngine.LAN_LOBBY:
                foc = lobby['focus']
                if foc == 'ip':
                    lobby['ip'] = handle_text_input(lobby['ip'], key, max_len=15)
                # Tab cycles focus
                if key == 9:
                    cycle = ['name'] if lobby['mode'] == 'host' else ['name', 'ip']
                    idx = cycle.index(foc) if foc in cycle else 0
                    lobby['focus'] = cycle[(idx + 1) % len(cycle)]

        # ── LAN WAITING ───────────────────────────────────────────────────
        elif state == GameEngine.LAN_WAITING:
            with lan_st.lock:
                joined = lan_st.joined
                expected = lan_st.expected
                ready_count = lan_st.ready
                local_ready = lan_st.local_ready
                started = lan_st.started
                err = lan_st.error

            if err:
                lobby['error'] = err
                engine.state['state'] = GameEngine.LAN_LOBBY
                if _announcer is not None:
                    _announcer.stop()
                    _announcer = None
                lan_st = None
            elif started:
                if _announcer is not None:
                    _announcer.stop()
                    _announcer = None
                engine.reset()
                engine.state['state'] = GameEngine.COUNTDOWN
                engine.state['countdown_start'] = now
            else:
                if clicked == 'ready' and not local_ready:
                    lan_st.set_local_ready()
                    local_ready = True
                    ready_count += 1
                buttons = renderer.draw_lan_waiting(
                    frame, joined, expected, ready_count, lobby['name'], lobby['mode'], local_ready
                )

        # ── COUNTDOWN ────────────────────────────────────────────────────
        elif state == GameEngine.COUNTDOWN:
            elapsed = engine.update_countdown(now, fw, fh)
            renderer.draw_countdown(frame, max(0, 3 - int(elapsed)))

        # ── PLAYING ────────────────────────────────────────────────────────────────
        elif state == GameEngine.PLAYING:
            all_hands = detector.detect_all(frame)
            play_info = engine.update_playing(now, fw, all_hands)
            targets = engine.state['targets']
            cleared = engine.state['cleared']

            for i, target in enumerate(targets):
                renderer.draw_target(frame, target,
                                     hold_progress=play_info['hold_progresses'][i],
                                     target_duration=play_info['target_duration'])
                if play_info['in_targets'][i]:
                    cv2.circle(frame, (target['x'], target['y']),
                               target['r'], (255, 255, 255), 2)
                if cleared[i]:
                    cv2.putText(frame, 'OK', (target['x'] - 18, target['y'] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 3, cv2.LINE_AA)
            renderer.draw_hud(frame, engine.state['score'],
                              engine.state['round_idx'] + 1,
                              engine.state['last_det_ges'],
                              perfect_streak=engine.state['perfect_streak'])
            if play_info['hand_w'] is not None:
                cv2.putText(frame, f"w:{play_info['hand_w']}",
                            (10, game_cfg.hud_h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame,
                            f"min:{play_info['min_w']} max:{play_info['max_w']}",
                            (10, game_cfg.hud_h + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        # ── RESULT ───────────────────────────────────────────────────────
        elif state == GameEngine.RESULT:
            for target in engine.state['targets']:
                renderer.draw_target(frame, target)
            renderer.draw_hud(frame, engine.state['score'],
                              engine.state['round_idx'] + 1,
                              engine.state['last_det_ges'],
                              perfect_streak=engine.state['perfect_streak'])
            renderer.draw_result(frame,
                                 hit_count=engine.state['last_round_hits'],
                                 hit_total=len(engine.state['targets']),
                                 round_score=engine.state['last_round_score'])
            engine.update_result(now, fw, fh)
        # ── GAMEOVER (single-player or LAN intermediate) ─────────────────
        elif state == GameEngine.GAMEOVER:
            if lan_st is not None:
                # LAN match: report score to background thread, wait for leaderboard
                with lan_st.lock:
                    lb = lan_st.leaderboard
                if lb is None:
                    # Score not yet sent — send once
                    if not lan_st._score_event.is_set():
                        lan_st.set_score(engine.state['score'])
                    renderer.draw_gameover(frame, engine.state['score'])
                else:
                    engine.state['state'] = GameEngine.LAN_LEADERBOARD
            else:
                buttons = renderer.draw_gameover(frame, engine.state['score'])
                if clicked == 'menu':
                    engine.state['state'] = GameEngine.MENU

        # ── LAN LEADERBOARD ───────────────────────────────────────────────
        elif state == GameEngine.LAN_LEADERBOARD:
            with lan_st.lock:
                lb = lan_st.leaderboard or []
            buttons = renderer.draw_lan_leaderboard(frame, lb, lobby['name'])
            if clicked == 'menu':
                if _announcer is not None:
                    _announcer.stop()
                    _announcer = None
                lan_st = None
                _server = None
                _client = None
                engine.state['state'] = GameEngine.MENU

        input_mgr.set_buttons(buttons)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        if key == ord('r') and state == GameEngine.GAMEOVER and lan_st is None:
            engine.reset()
            engine.start_countdown(now)
        if clicked == 'restart' and state == GameEngine.GAMEOVER and lan_st is None:
            engine.reset()
            engine.start_countdown(now)

    cap.release()
    cv2.destroyAllWindows()
    scanner.stop()
    if _announcer is not None:
        _announcer.stop()
    print(f"Final score: {engine.state['score']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    run(GameConfig(), GestureConfig(), LanConfig())


if __name__ == '__main__':
    main()
