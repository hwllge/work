import json
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple


def _send_json(sock: socket.socket, payload: dict):
    data = (json.dumps(payload, ensure_ascii=True) + '\n').encode('utf-8')
    sock.sendall(data)


def _recv_json(reader) -> Optional[dict]:
    line = reader.readline()
    if not line:
        return None
    return json.loads(line.decode('utf-8').strip())


class LanServer:
    def __init__(self, host: str, port: int, expected_clients: int = 3):
        self.host = host
        self.port = port
        self.expected_clients = expected_clients
        self.server_sock: Optional[socket.socket] = None
        self.clients: List[Tuple[socket.socket, object, str]] = []

    def start(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(self.expected_clients)

    def _accept_one_join(self):
        if self.server_sock is None:
            raise RuntimeError('Server is not started.')

        conn, addr = self.server_sock.accept()
        reader = conn.makefile('rb')
        msg = _recv_json(reader)
        if not msg or msg.get('type') != 'join':
            _send_json(conn, {'type': 'join_ack', 'ok': False, 'reason': 'invalid_join'})
            conn.close()
            return False

        name = msg.get('name') or f'client_{len(self.clients) + 1}'
        self.clients.append((conn, reader, name))
        _send_json(
            conn,
            {
                'type': 'join_ack',
                'ok': True,
                'player': name,
                'joined': len(self.clients) + 1,
                # do not force full room; expected is current joined count (host included)
                'expected': len(self.clients) + 1,
            },
        )
        print(f'[LAN] Joined {name} from {addr} ({len(self.clients)} clients)')
        return True

    def send_start(self):
        for conn, _, _ in self.clients:
            _send_json(conn, {'type': 'start'})
        print('[LAN] Match started.')

    def wait_all_ready(self, host_ready_event: threading.Event, on_progress=None):
        if self.server_sock is None:
            raise RuntimeError('Server is not started.')

        self.server_sock.settimeout(0.2)
        ready_clients = set()
        host_ready = False

        while True:
            try:
                joined = self._accept_one_join()
                if joined:
                    conn, _, _ = self.clients[-1]
                    conn.settimeout(0.2)
            except socket.timeout:
                pass

            expected_total = len(self.clients) + 1

            if not host_ready and host_ready_event.is_set():
                host_ready = True
                if on_progress is not None:
                    on_progress((1 if host_ready else 0) + len(ready_clients), expected_total)

            for conn, reader, name in self.clients:
                if name in ready_clients:
                    continue
                try:
                    msg = _recv_json(reader)
                except socket.timeout:
                    continue
                except Exception:
                    continue

                if not msg:
                    continue
                if msg.get('type') == 'ready' and bool(msg.get('ready', True)):
                    ready_clients.add(name)
                    if on_progress is not None:
                        on_progress((1 if host_ready else 0) + len(ready_clients), expected_total)

            total_ready = (1 if host_ready else 0) + len(ready_clients)
            if total_ready >= expected_total:
                return

            time.sleep(0.05)

    def collect_scores(self) -> List[Dict[str, int]]:
        results: List[Dict[str, int]] = []
        for conn, reader, name in self.clients:
            try:
                msg = _recv_json(reader)
                score = 0
                if msg and msg.get('type') == 'score':
                    score = int(msg.get('score', 0))
                results.append({'name': name, 'score': score})
            except Exception:
                results.append({'name': name, 'score': 0})

        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def broadcast_leaderboard(self, leaderboard: List[Dict[str, int]]):
        payload = {'type': 'leaderboard', 'leaderboard': leaderboard}
        for conn, _, _ in self.clients:
            try:
                _send_json(conn, payload)
            except Exception:
                pass

    def close(self):
        for conn, reader, _ in self.clients:
            try:
                reader.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        self.clients.clear()

        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None


class LanClient:
    def __init__(self, server_ip: str, port: int, player_name: str):
        self.server_ip = server_ip
        self.port = port
        self.player_name = player_name
        self.sock: Optional[socket.socket] = None
        self.reader = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_ip, self.port))
        self.reader = self.sock.makefile('rb')

    def join(self):
        if self.sock is None or self.reader is None:
            raise RuntimeError('Client is not connected.')

        _send_json(
            self.sock,
            {
                'type': 'join',
                'name': self.player_name,
            },
        )
        ack = _recv_json(self.reader)
        if not ack or not ack.get('ok'):
            reason = ack.get('reason') if ack else 'no_response'
            raise RuntimeError(f'Join failed: {reason}')
        return ack

    def wait_start(self):
        if self.reader is None:
            raise RuntimeError('Client is not connected.')
        msg = _recv_json(self.reader)
        if not msg or msg.get('type') != 'start':
            raise RuntimeError('Server did not send start signal.')

    def send_ready(self):
        if self.sock is None:
            raise RuntimeError('Client is not connected.')
        _send_json(self.sock, {'type': 'ready', 'ready': True})

    def send_score_and_get_leaderboard(self, score: int):
        if self.sock is None or self.reader is None:
            raise RuntimeError('Client is not connected.')

        _send_json(self.sock, {'type': 'score', 'score': int(score)})
        msg = _recv_json(self.reader)
        if not msg or msg.get('type') != 'leaderboard':
            raise RuntimeError('Server did not send leaderboard.')
        return msg.get('leaderboard', [])

    def close(self):
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None

        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


def get_local_ip() -> str:
    """Best-effort local LAN IP discovery."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


class LanRoomAnnouncer:
    def __init__(self, room_name: str, host: str, port: int, discovery_port: int, max_players: int):
        self.room_name = room_name
        self.host = host
        self.port = port
        self.discovery_port = discovery_port
        self.max_players = max_players
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            payload = {
                'type': 'room_announce',
                'name': self.room_name,
                'host': self.host,
                'port': self.port,
                'max_players': self.max_players,
            }
            data = json.dumps(payload, ensure_ascii=True).encode('utf-8')
            while not self._stop_event.is_set():
                sock.sendto(data, ('255.255.255.255', self.discovery_port))
                time.sleep(1.0)
        finally:
            sock.close()

    def stop(self):
        self._stop_event.set()


class LanRoomScanner:
    def __init__(self, discovery_port: int):
        self.discovery_port = discovery_port
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._rooms = {}

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', self.discovery_port))
            sock.settimeout(1.0)

            while not self._stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(2048)
                except socket.timeout:
                    self._evict_stale()
                    continue
                except Exception:
                    continue

                try:
                    msg = json.loads(data.decode('utf-8'))
                except Exception:
                    continue

                if msg.get('type') != 'room_announce':
                    continue

                host = msg.get('host')
                port = msg.get('port')
                if not host or not port:
                    continue

                key = f'{host}:{port}'
                with self._lock:
                    self._rooms[key] = {
                        'name': msg.get('name', 'Room'),
                        'host': host,
                        'port': int(port),
                        'max_players': int(msg.get('max_players', 4)),
                        '_seen': time.time(),
                    }
        finally:
            sock.close()

    def _evict_stale(self):
        now = time.time()
        with self._lock:
            stale_keys = [k for k, v in self._rooms.items() if now - v['_seen'] > 3.5]
            for k in stale_keys:
                self._rooms.pop(k, None)

    def get_rooms(self) -> List[dict]:
        self._evict_stale()
        with self._lock:
            rooms = []
            for room in self._rooms.values():
                rooms.append({
                    'name': room['name'],
                    'host': room['host'],
                    'port': room['port'],
                    'max_players': room['max_players'],
                })
            rooms.sort(key=lambda r: (r['name'], r['host']))
            return rooms

    def stop(self):
        self._stop_event.set()
