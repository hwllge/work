import argparse
import os
import socket
import threading
import time
from collections import deque
from glob import glob

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

from config import GameConfig, GestureConfig, LanConfig
from game_engine import GameEngine
from gesture_detector import GestureDetector
from input_manager import InputManager, handle_text_input
from lan import LanClient, LanRoomAnnouncer, LanRoomScanner, LanServer, get_local_ip
from renderer import GameRenderer


# ── Shared LAN state (same behavior as src.py) ──────────────────────────────
class LanState:
    def __init__(self):
        self.lock = threading.Lock()
        self.joined = 0
        self.expected = 1
        self.ready = 0
        self.local_ready = False
        self.started = False
        self.leaderboard = None
        self.error = None
        self._ready_event = threading.Event()
        self._score_event = threading.Event()
        self._cancel_event = threading.Event()
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


def _open_camera(game_cfg):
    """Try multiple camera sources and return the first working capture."""
    candidates = []

    env_idx = os.environ.get('CAMERA_INDEX')
    env_dev = os.environ.get('CAMERA_DEVICE')

    # If device path is explicitly provided, only try that device.
    if env_dev:
        candidates = [env_dev]
    elif env_idx is not None:
        try:
            pref_idx = int(env_idx)
            # Keep fallback minimal to avoid long /dev/video* timeout chains.
            candidates = [pref_idx, 0, 1, 2, 3]
        except ValueError:
            candidates = [0, 1, 2, 3]
    else:
        candidates = [0, 1, 2, 3]

    # Optional exhaustive scan for debugging only.
    if os.environ.get('CAMERA_SCAN_ALL') == '1':
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


def _server_thread(server: LanServer, lan_st: LanState, start_delay_s: float):
    try:
        def _on_ready_progress(joined_count: int, ready_count: int, expected: int):
            with lan_st.lock:
                lan_st.joined = joined_count
                lan_st.ready = ready_count
                lan_st.expected = expected

        server.wait_all_ready(
            lan_st._ready_event,
            on_progress=_on_ready_progress,
            stop_event=lan_st._cancel_event,
        )
        if lan_st._cancel_event.is_set():
            return

        if start_delay_s > 0:
            wait_until = time.time() + start_delay_s
            while time.time() < wait_until:
                if lan_st._cancel_event.is_set():
                    return
                time.sleep(0.05)

        server.send_start()
        with lan_st.lock:
            lan_st.started = True

        score = lan_st.wait_score()
        leaderboard = server.collect_scores()
        leaderboard.append({'name': '_host_', 'score': score})
        leaderboard.sort(key=lambda x: x['score'], reverse=True)
        server.broadcast_leaderboard(leaderboard)
        with lan_st.lock:
            lan_st.leaderboard = leaderboard
    except Exception as e:
        if not lan_st._cancel_event.is_set():
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

        def _on_ready_state(joined_count: int, ready_count: int, expected: int):
            with lan_st.lock:
                lan_st.joined = joined_count
                lan_st.ready = ready_count
                lan_st.expected = expected

        client.wait_start(
            ready_event=lan_st._ready_event,
            on_ready_state=_on_ready_state,
            stop_event=lan_st._cancel_event,
        )
        if lan_st._cancel_event.is_set():
            return
        with lan_st.lock:
            lan_st.started = True

        score = lan_st.wait_score()
        lb = client.send_score_and_get_leaderboard(score)
        with lan_st.lock:
            lan_st.leaderboard = lb
    except Exception as e:
        if not lan_st._cancel_event.is_set():
            with lan_st.lock:
                lan_st.error = str(e)
    finally:
        client.close()


class FrameBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_jpg = None

    def set_frame(self, frame):
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        with self._lock:
            self._latest_jpg = encoded.tobytes()

    def get_frame(self):
        with self._lock:
            return self._latest_jpg


class WebInput:
    def __init__(self):
        self._lock = threading.Lock()
        self._key_queue = deque()

    def push_key(self, key_code: int):
        with self._lock:
            self._key_queue.append(key_code)

    def pop_key(self) -> int:
        with self._lock:
            if self._key_queue:
                return self._key_queue.popleft()
        return -1


def _map_browser_key_to_waitkey(key_name: str) -> int:
    if not key_name:
        return -1
    k = key_name.lower()
    if k == 'backspace':
        return 8
    if k == 'tab':
        return 9
    if len(k) == 1:
        return ord(k)
    return -1


class WebGameApp:
    def __init__(self, game_cfg, ges_cfg, lan_cfg):
        self.game_cfg = game_cfg
        self.ges_cfg = ges_cfg
        self.lan_cfg = lan_cfg

        self.renderer = GameRenderer(game_cfg, ges_cfg)
        self.engine = GameEngine(game_cfg, ges_cfg.gestures)
        self.detector = GestureDetector(game_cfg, ges_cfg)
        self.input_mgr = InputManager()
        self.frame_bus = FrameBus()
        self.web_input = WebInput()

        self._running = threading.Event()
        self._running.set()

        self.cap = None
        self.scanner = None
        self._announcer = None
        self._server = None
        self._client = None
        self._lan_st = None

        self.lobby = {
            'mode': None,
            'focus': 'name',
            'name': LAN_NICKNAMES[0],
            'name_picker_open': False,
            'name_options': LAN_NICKNAMES,
            'ip': '',
            'rooms': [],
            'error': None,
        }

    def click(self, x: int, y: int):
        self.input_mgr.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, None, None)

    def key(self, key_name: str):
        key_code = _map_browser_key_to_waitkey(key_name)
        if key_code != -1:
            self.web_input.push_key(key_code)

    def stop(self):
        self._running.clear()

    def run_loop(self):
        self.cap, opened_src = _open_camera(self.game_cfg)
        if self.cap is None:
            err = self._error_frame('Camera open failed')
            self.frame_bus.set_frame(err)
            return

        print(f'[INFO] Camera source: {opened_src}')

        self.scanner = LanRoomScanner(self.lan_cfg.discovery_port)
        self.scanner.start()

        try:
            while self._running.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                if frame.shape[1] != self.game_cfg.cam_w or frame.shape[0] != self.game_cfg.cam_h:
                    frame = cv2.resize(frame, (self.game_cfg.cam_w, self.game_cfg.cam_h))

                now = time.time()
                fh, fw = frame.shape[:2]
                clicked = self.input_mgr.consume_click()
                key = self.web_input.pop_key()
                state = self.engine.state['state']
                buttons = {}

                if state == GameEngine.MENU:
                    buttons = self.renderer.draw_menu(frame)
                    if clicked == 'run':
                        self.engine.reset()
                        self.engine.start_countdown(now)
                    elif clicked == 'howto':
                        self.engine.state['state'] = GameEngine.HOWTO
                    elif clicked == 'lan':
                        self.engine.state['state'] = GameEngine.LAN_ROOM_LIST

                elif state == GameEngine.HOWTO:
                    buttons = self.renderer.draw_howto(frame)
                    if clicked == 'back':
                        self.engine.state['state'] = GameEngine.MENU

                elif state == GameEngine.LAN_ROOM_LIST:
                    self.lobby['rooms'] = self.scanner.get_rooms()
                    buttons = self.renderer.draw_lan_room_list(frame, self.lobby['rooms'])
                    if clicked == 'back':
                        self.engine.state['state'] = GameEngine.MENU
                    elif clicked == 'create_room':
                        self.lobby['mode'] = 'host'
                        self.lobby['focus'] = 'name'
                        self.lobby['error'] = None
                        self.engine.state['state'] = GameEngine.LAN_LOBBY
                    elif clicked == 'join_manual':
                        self.lobby['mode'] = 'join'
                        self.lobby['focus'] = 'ip'
                        self.lobby['error'] = None
                        self.engine.state['state'] = GameEngine.LAN_LOBBY
                    elif clicked and clicked.startswith('room_'):
                        try:
                            idx = int(clicked.split('_')[1])
                            room = self.lobby['rooms'][idx]
                            self.lobby['mode'] = 'join'
                            self.lobby['ip'] = room.get('host', '')
                            self.lobby['focus'] = 'name'
                            self.lobby['error'] = None
                            self.engine.state['state'] = GameEngine.LAN_LOBBY
                        except Exception:
                            pass

                elif state == GameEngine.LAN_LOBBY:
                    buttons = self.renderer.draw_lan_lobby(frame, self.lobby)

                    for fname in ('ip',):
                        if clicked == f'field_{fname}':
                            self.lobby['focus'] = fname

                    if clicked == 'field_name':
                        self.lobby['name_picker_open'] = not self.lobby.get('name_picker_open', False)
                        self.lobby['focus'] = 'name'

                    if clicked and clicked.startswith('name_opt_'):
                        try:
                            idx = int(clicked.split('_')[-1])
                            options = self.lobby.get('name_options', [])
                            if 0 <= idx < len(options):
                                self.lobby['name'] = options[idx]
                                self.lobby['name_picker_open'] = False
                        except Exception:
                            pass

                    if clicked == 'host':
                        self.lobby['mode'] = 'host'
                        self.lobby['name_picker_open'] = False
                    elif clicked == 'join':
                        self.lobby['mode'] = 'join'
                        self.lobby['name_picker_open'] = False
                    elif clicked == 'back':
                        self.engine.state['state'] = GameEngine.LAN_ROOM_LIST
                        self.lobby['error'] = None
                        self.lobby['name_picker_open'] = False
                    elif clicked == 'confirm' and self.lobby['mode']:
                        if not self.lobby['name'].strip():
                            self.lobby['error'] = 'Player name is required.'
                        elif self.lobby['mode'] == 'join' and not self.lobby['ip'].strip():
                            self.lobby['error'] = 'Server IP is required.'
                        else:
                            self.lobby['error'] = None
                            self._lan_st = LanState()
                            self._lan_st.expected = self.lan_cfg.expected_clients + 1

                            if self.lobby['mode'] == 'host':
                                self._server = LanServer(
                                    '0.0.0.0',
                                    self.lan_cfg.port,
                                    self.lan_cfg.expected_clients,
                                    self.lan_cfg.min_start_players,
                                )
                                self._server.start()
                                self._lan_st.joined = 1
                                self._announcer = LanRoomAnnouncer(
                                    room_name=f"{self.lobby['name']}'s room",
                                    host=get_local_ip(),
                                    port=self.lan_cfg.port,
                                    discovery_port=self.lan_cfg.discovery_port,
                                    max_players=self.lan_cfg.expected_clients + 1,
                                )
                                self._announcer.start()
                                t = threading.Thread(target=_server_thread,
                                                     args=(self._server, self._lan_st, self.lan_cfg.start_delay_s), daemon=True)
                                t.start()
                            else:
                                self._client = LanClient(self.lobby['ip'].strip(), self.lan_cfg.port, self.lobby['name'].strip())
                                try:
                                    self._client.connect()
                                    t = threading.Thread(target=_client_thread,
                                                         args=(self._client, self._lan_st), daemon=True)
                                    t.start()
                                except Exception as e:
                                    self.lobby['error'] = f'Cannot connect: {e}'
                                    self._lan_st = None
                                    self._client = None

                            if self._lan_st is not None:
                                self.engine.state['state'] = GameEngine.LAN_WAITING

                    foc = self.lobby['focus']
                    if foc == 'ip':
                        self.lobby['ip'] = handle_text_input(self.lobby['ip'], key, max_len=15)
                    if key == 9:
                        cycle = ['name'] if self.lobby['mode'] == 'host' else ['name', 'ip']
                        idx = cycle.index(foc) if foc in cycle else 0
                        self.lobby['focus'] = cycle[(idx + 1) % len(cycle)]

                elif state == GameEngine.LAN_WAITING:
                    with self._lan_st.lock:
                        joined = self._lan_st.joined
                        expected = self._lan_st.expected
                        ready_count = self._lan_st.ready
                        local_ready = self._lan_st.local_ready
                        started = self._lan_st.started
                        err = self._lan_st.error

                    if clicked == 'back_waiting':
                        self._lan_st._cancel_event.set()
                        if self._announcer is not None:
                            self._announcer.stop()
                            self._announcer = None
                        if self._client is not None:
                            self._client.close()
                            self._client = None
                        if self._server is not None:
                            self._server.close()
                            self._server = None
                        self._lan_st = None
                        self.lobby['error'] = None
                        self.engine.state['state'] = GameEngine.LAN_ROOM_LIST
                        continue

                    if err:
                        self.lobby['error'] = err
                        self.engine.state['state'] = GameEngine.LAN_LOBBY
                        if self._announcer is not None:
                            self._announcer.stop()
                            self._announcer = None
                        self._lan_st = None
                    elif started:
                        if self._announcer is not None:
                            self._announcer.stop()
                            self._announcer = None
                        self.engine.reset()
                        self.engine.state['state'] = GameEngine.COUNTDOWN
                        self.engine.state['countdown_start'] = now
                    else:
                        if clicked == 'ready' and not local_ready:
                            self._lan_st.set_local_ready()
                            local_ready = True
                            ready_count += 1
                        buttons = self.renderer.draw_lan_waiting(
                            frame, joined, expected, ready_count,
                            self.lobby['name'], self.lobby['mode'], local_ready
                        )

                elif state == GameEngine.COUNTDOWN:
                    elapsed = self.engine.update_countdown(now, fw, fh)
                    self.renderer.draw_countdown(frame, max(0, 3 - int(elapsed)))

                elif state == GameEngine.PLAYING:
                    all_hands = self.detector.detect_all(frame)
                    play_info = self.engine.update_playing(now, fw, all_hands)
                    targets = self.engine.state['targets']
                    cleared = self.engine.state['cleared']

                    for i, target in enumerate(targets):
                        self.renderer.draw_target(
                            frame,
                            target,
                            hold_progress=play_info['hold_progresses'][i],
                            target_duration=play_info['target_duration'],
                        )
                        if play_info['in_targets'][i]:
                            cv2.circle(frame, (target['x'], target['y']),
                                       target['r'], (255, 255, 255), 2)
                        if cleared[i]:
                            cv2.putText(frame, 'OK', (target['x'] - 18, target['y'] + 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 3, cv2.LINE_AA)

                    self.renderer.draw_hud(
                        frame,
                        self.engine.state['score'],
                        self.engine.state['round_idx'] + 1,
                        self.engine.state['last_det_ges'],
                        perfect_streak=self.engine.state['perfect_streak'],
                    )

                    if play_info['hand_w'] is not None:
                        cv2.putText(frame, f"w:{play_info['hand_w']}",
                                    (10, self.game_cfg.hud_h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        cv2.putText(frame,
                                    f"min:{play_info['min_w']} max:{play_info['max_w']}",
                                    (10, self.game_cfg.hud_h + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                elif state == GameEngine.RESULT:
                    for target in self.engine.state['targets']:
                        self.renderer.draw_target(frame, target)
                    self.renderer.draw_hud(
                        frame,
                        self.engine.state['score'],
                        self.engine.state['round_idx'] + 1,
                        self.engine.state['last_det_ges'],
                        perfect_streak=self.engine.state['perfect_streak'],
                    )
                    self.renderer.draw_result(
                        frame,
                        hit_count=self.engine.state['last_round_hits'],
                        hit_total=len(self.engine.state['targets']),
                        round_score=self.engine.state['last_round_score'],
                    )
                    self.engine.update_result(now, fw, fh)

                elif state == GameEngine.GAMEOVER:
                    if self._lan_st is not None:
                        with self._lan_st.lock:
                            lb = self._lan_st.leaderboard
                        if lb is None:
                            if not self._lan_st._score_event.is_set():
                                self._lan_st.set_score(self.engine.state['score'])
                            self.renderer.draw_gameover(frame, self.engine.state['score'])
                        else:
                            self.engine.state['state'] = GameEngine.LAN_LEADERBOARD
                    else:
                        buttons = self.renderer.draw_gameover(frame, self.engine.state['score'])
                        if clicked == 'menu':
                            self.engine.state['state'] = GameEngine.MENU

                elif state == GameEngine.LAN_LEADERBOARD:
                    with self._lan_st.lock:
                        lb = self._lan_st.leaderboard or []
                    buttons = self.renderer.draw_lan_leaderboard(frame, lb, self.lobby['name'])
                    if clicked == 'menu':
                        if self._announcer is not None:
                            self._announcer.stop()
                            self._announcer = None
                        self._lan_st = None
                        self._server = None
                        self._client = None
                        self.engine.state['state'] = GameEngine.MENU

                self.input_mgr.set_buttons(buttons)
                if key == ord('q'):
                    self.stop()
                if key == ord('r') and state == GameEngine.GAMEOVER and self._lan_st is None:
                    self.engine.reset()
                    self.engine.start_countdown(now)
                if clicked == 'restart' and state == GameEngine.GAMEOVER and self._lan_st is None:
                    self.engine.reset()
                    self.engine.start_countdown(now)

                self.frame_bus.set_frame(frame)
                time.sleep(0.001)
        finally:
            if self.cap is not None:
                self.cap.release()
            if self.scanner is not None:
                self.scanner.stop()
            if self._announcer is not None:
                self._announcer.stop()
            print(f"Final score: {self.engine.state['score']}")

    def _error_frame(self, msg: str):
        frame = 255 * np.ones((self.game_cfg.cam_h, self.game_cfg.cam_w, 3), dtype='uint8')
        cv2.putText(frame, msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return frame


HTML_PAGE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RPS Target Game - Web</title>
  <style>
    :root {
      --bg1: #0b1020;
      --bg2: #13243f;
      --card: rgba(8, 12, 24, 0.8);
      --line: rgba(255, 255, 255, 0.2);
      --text: #f6fbff;
      --accent: #35d18e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(1200px 600px at 15% 10%, #244a7e 0%, transparent 70%),
        radial-gradient(900px 500px at 85% 95%, #1c6d54 0%, transparent 65%),
        linear-gradient(145deg, var(--bg1), var(--bg2));
      color: var(--text);
      font-family: "Noto Sans KR", "Segoe UI", sans-serif;
    }
    .wrap {
      width: min(98vw, 1180px);
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      backdrop-filter: blur(8px);
    }
    .head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
      font-size: 14px;
    }
    .kbd {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 2px 8px;
      margin: 0 3px;
      background: rgba(255,255,255,0.06);
    }
    .viewport {
      width: 100%;
      aspect-ratio: 1024 / 600;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.18);
      background: #000;
    }
    #feed {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
      cursor: crosshair;
      user-select: none;
      -webkit-user-drag: none;
    }
    .tail {
      margin-top: 10px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 8px 12px;
      background: #1e2f55;
      color: #fff;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.12); }
    .q { background: #b44d4d; }
    .r { background: var(--accent); color: #0b1a14; font-weight: 700; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div>웹 화면 클릭으로 버튼을 누를 수 있습니다.</div>
      <div><span class="kbd">Q</span> 종료 <span class="kbd">R</span> 재시작</div>
    </div>
    <div class="viewport">
      <img id="feed" src="/video_feed" alt="RPS stream" />
    </div>
    <div class="tail">
      <button class="r" id="btn-r">R 보내기</button>
      <button class="q" id="btn-q">Q 보내기</button>
    </div>
  </div>

  <script>
    const CAM_W = 1024;
    const CAM_H = 600;
    const feed = document.getElementById('feed');

    async function sendKey(k) {
      await fetch('/key', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ key: k })
      });
    }

    feed.addEventListener('click', async (e) => {
      const rect = feed.getBoundingClientRect();
      const x = Math.max(0, Math.min(CAM_W - 1, Math.round((e.clientX - rect.left) * CAM_W / rect.width)));
      const y = Math.max(0, Math.min(CAM_H - 1, Math.round((e.clientY - rect.top) * CAM_H / rect.height)));
      await fetch('/click', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ x, y })
      });
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'q' || e.key === 'Q') {
        e.preventDefault();
        sendKey('q');
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        sendKey('r');
      } else if (e.key === 'Tab') {
        e.preventDefault();
        sendKey('Tab');
      } else if (e.key === 'Backspace') {
        e.preventDefault();
        sendKey('Backspace');
      } else if (e.key.length === 1) {
        sendKey(e.key);
      }
    });

    document.getElementById('btn-q').addEventListener('click', () => sendKey('q'));
    document.getElementById('btn-r').addEventListener('click', () => sendKey('r'));
  </script>
</body>
</html>
"""


def create_app(game_cfg=None, ges_cfg=None, lan_cfg=None):
    game_cfg = game_cfg or GameConfig()
    ges_cfg = ges_cfg or GestureConfig()
    lan_cfg = lan_cfg or LanConfig()

    app = Flask(__name__)
    web_game = WebGameApp(game_cfg, ges_cfg, lan_cfg)

    game_thread = threading.Thread(target=web_game.run_loop, daemon=True)
    game_thread.start()

    @app.route('/')
    def index():
        return render_template_string(HTML_PAGE)

    @app.route('/video_feed')
    def video_feed():
        def gen():
            while web_game._running.is_set():
                frame = web_game.frame_bus.get_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)

        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/click', methods=['POST'])
    def click():
        data = request.get_json(silent=True) or {}
        x = int(data.get('x', -1))
        y = int(data.get('y', -1))
        if 0 <= x < game_cfg.cam_w and 0 <= y < game_cfg.cam_h:
            web_game.click(x, y)
            return jsonify({'ok': True})
        return jsonify({'ok': False, 'error': 'invalid coordinates'}), 400

    @app.route('/key', methods=['POST'])
    def key():
        data = request.get_json(silent=True) or {}
        key_name = str(data.get('key', ''))
        web_game.key(key_name)
        return jsonify({'ok': True})

    return app


def _get_bind_ip(host: str):
    if host and host != '0.0.0.0':
        return host
    try:
        return get_local_ip()
    except (OSError, socket.error):
        return '127.0.0.1'


def main():
    parser = argparse.ArgumentParser(description='Run miniproject as a web game (MJPEG stream).')
    parser.add_argument('--host', default='0.0.0.0', help='bind host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='bind port (default: 8080)')
    parser.add_argument('--camera-index', type=int, default=0,
                        help='preferred camera index (default: 0, usually laptop built-in)')
    parser.add_argument('--camera-device', default='',
                        help='preferred camera device path, e.g. /dev/video0')
    args = parser.parse_args()

    # Apply camera preference before the game thread opens a camera.
    if args.camera_device:
        os.environ['CAMERA_DEVICE'] = args.camera_device
    else:
        os.environ['CAMERA_INDEX'] = str(args.camera_index)

    app = create_app(GameConfig(), GestureConfig(), LanConfig())
    bind_ip = _get_bind_ip(args.host)

    print(f'[INFO] Web game URL: http://{bind_ip}:{args.port}')
    if args.camera_device:
        print(f'[INFO] Preferred camera device: {args.camera_device}')
    else:
        print(f'[INFO] Preferred camera index: {args.camera_index}')
    print('[INFO] Browser click -> game click, keyboard -> game key input')
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
