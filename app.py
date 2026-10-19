"""Train Queue: single-user local GPU queue, no third-party dependencies."""
import argparse
import csv
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import secrets
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, parse_qs

from worker import atomic_json, group_alive, process_start, same_process, stop_group
from process_monitor import ProcessMonitor

ROOT = Path(__file__).resolve().parent
TERMINAL = {'succeeded', 'failed', 'canceled', 'interrupted'}


def inspect_gpus():
    try:
        info = subprocess.run([
            'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
        ], capture_output=True, text=True, timeout=4, check=True)
        apps = subprocess.run([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits',
        ], capture_output=True, text=True, timeout=4, check=True)
        busy = {r[0].strip() for r in csv.reader(io.StringIO(apps.stdout)) if r}
        gpus = []
        for row in csv.reader(io.StringIO(info.stdout)):
            index, uuid, name, total, used, util = [x.strip() for x in row]
            gpus.append(dict(index=index, uuid=uuid, name=name, total_mb=int(total),
                             used_mb=int(used), utilization=int(util), busy=uuid in busy))
        if not gpus:
            return [], '未检测到 NVIDIA GPU'
        return gpus, None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        detail = (getattr(exc, 'stderr', None) or str(exc)).strip()
        return [], 'GPU 检测不可用：' + detail[:400]


def normalize_submission(data, default_cwd):
    command = data.get('command', '')
    if not isinstance(command, str) or not command.strip() or len(command) > 65536:
        raise ValueError('请输入训练命令，最长 65536 字符')
    command = command.strip()
    gpu = str(data.get('gpu', 'auto')).strip()
    # Common pasted shell prefixes retain their physical GPU selection.
    prefix = re.match(r'^(?:export\s+)?CUDA_VISIBLE_DEVICES=([\d,]+)(?:\s*;\s*|\s*\n\s*|\s+)', command)
    if prefix:
        pasted_gpu = prefix.group(1)
        if gpu not in ('auto', pasted_gpu):
            raise ValueError('命令内的 GPU 编号与选择器冲突，请统一设置')
        gpu = pasted_gpu
        command = command[prefix.end():].strip()
    if re.search(r'\bCUDA_VISIBLE_DEVICES\s*=', command):
        raise ValueError('请在 GPU 选择器中指定显卡；也支持命令开头的 CUDA_VISIBLE_DEVICES=0 前缀')
    if not command:
        raise ValueError('GPU 设置后还需要一条训练命令')
    if gpu not in ('auto', 'cpu') and not re.fullmatch(r'\d+(,\d+)*', gpu):
        raise ValueError('GPU 请填写 auto、cpu 或物理编号（如 0 或 0,1）')
    if gpu not in ('auto', 'cpu'):
        ids = [str(int(i)) for i in gpu.split(',')]
        if len(set(ids)) != len(ids):
            raise ValueError('GPU 编号不能重复')
        gpu = ','.join(ids)
    cwd = Path(str(data.get('cwd') or default_cwd)).expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError(f'工作目录不存在：{cwd}')
    try:
        memory = int(data.get('min_free_mb', 1024))
        if memory < 0 or memory > 1000000:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError('所需空闲显存必须在 0～1000000 MiB 之间')
    return command, str(cwd), gpu, memory


class Queue:
    def __init__(self, data_dir, default_cwd, gpu_probe=inspect_gpus):
        self.directory = Path(data_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_file = (self.directory / 'queue.lock').open('a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            raise RuntimeError('同一数据目录已有服务运行，不能重复启动')
        self.db = sqlite3.connect(self.directory / 'queue.sqlite3', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
          PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, command TEXT NOT NULL, cwd TEXT NOT NULL,
            gpu TEXT NOT NULL, min_free_mb INTEGER NOT NULL, assigned TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'queued', created_at REAL NOT NULL, started_at REAL,
            finished_at REAL, exit_code INTEGER, pid INTEGER, process_start TEXT,
            note TEXT NOT NULL DEFAULT '', position INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          INSERT OR IGNORE INTO settings VALUES ('paused', 'false');
          INSERT OR IGNORE INTO settings VALUES ('stop_on_failure', 'false');
        ''')
        self.default_cwd = str(Path(default_cwd).resolve())
        self.gpu_probe = gpu_probe
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.wakeup = threading.Event()
        self.children = {}
        self.gpus = []
        self.gpu_error = '正在检测 GPU…'
        self.last_probe = 0
        self.scheduler_error = None
        self.thread = None
        self.process_monitor = ProcessMonitor(self.directory)

    def external_scope(self):
        with self.lock:
            return [r['pid'] for r in self.db.execute(
                "SELECT pid FROM tasks WHERE status IN ('starting','running','canceling')") if r['pid']]

    def external_state(self, mode='training'):
        with self.lock:
            gpu_names = {g['uuid']:g['index'] for g in self.gpus}
        return self.process_monitor.snapshot(self.external_scope(),mode,gpu_names=gpu_names)

    def task_dir(self, task_id):
        return self.directory / 'tasks' / str(task_id)

    def rows(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM tasks ORDER BY position,id')]

    def settings(self):
        return {r['key']: json.loads(r['value']) for r in self.db.execute('SELECT * FROM settings')}

    def add(self, data):
        command, cwd, gpu, memory = normalize_submission(data, self.default_cwd)
        with self.lock, self.db:
            position = self.db.execute('SELECT COALESCE(MAX(position),0)+1 FROM tasks').fetchone()[0]
            cur = self.db.execute('''INSERT INTO tasks
              (command,cwd,gpu,min_free_mb,created_at,position) VALUES (?,?,?,?,?,?)''',
              (command, cwd, gpu, memory, time.time(), position))
            task_id = cur.lastrowid
        self.wakeup.set()
        return task_id

    def action(self, task_id, action):
        with self.lock, self.db:
            task = self.db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            if not task:
                raise ValueError('任务不存在')
            if action == 'cancel':
                if task['status'] == 'queued':
                    self.db.execute("UPDATE tasks SET status='canceled',finished_at=?,note='' WHERE id=?", (time.time(), task_id))
                elif task['status'] in ('starting', 'running', 'canceling'):
                    (self.task_dir(task_id) / 'cancel').touch()
                    self.db.execute("UPDATE tasks SET status='canceling',note='正在终止训练进程…' WHERE id=?", (task_id,))
                else:
                    raise ValueError('任务已经结束')
            elif action == 'retry':
                if task['status'] not in TERMINAL:
                    raise ValueError('只能重新排队已结束的任务')
                return self.add(dict(task))
            elif action == 'top':
                if task['status'] != 'queued':
                    raise ValueError('只能调整等待中的任务')
                position = self.db.execute('SELECT COALESCE(MIN(position),0)-1 FROM tasks').fetchone()[0]
                self.db.execute('UPDATE tasks SET position=? WHERE id=?', (position, task_id))
            else:
                raise ValueError('未知操作')
        self.wakeup.set()
        return task_id

    def update_settings(self, data):
        with self.lock, self.db:
            for key in ('paused', 'stop_on_failure'):
                if key in data:
                    if not isinstance(data[key], bool):
                        raise ValueError('设置值必须为布尔值')
                    self.db.execute('UPDATE settings SET value=? WHERE key=?', (json.dumps(data[key]), key))
        self.wakeup.set()

    def state(self):
        with self.lock:
            tasks = self.rows()
            for task in tasks:
                task['assigned'] = json.loads(task['assigned'])
                task.pop('process_start', None)
            return dict(tasks=tasks, settings=self.settings(), gpus=self.gpus,
                        gpu_error=self.gpu_error, scheduler_error=self.scheduler_error,
                        default_cwd=self.default_cwd)

    def log(self, task_id, offset=0):
        if offset < 0:
            raise ValueError('日志偏移不能为负数')
        with self.lock:
            if not self.db.execute('SELECT 1 FROM tasks WHERE id=?', (task_id,)).fetchone():
                raise ValueError('任务不存在')
        path = self.task_dir(task_id) / 'output.log'
        if not path.exists():
            return dict(text='', offset=0, truncated=False)
        with path.open('rb') as stream:
            size = os.fstat(stream.fileno()).st_size
            start = max(min(offset, size), size - 256 * 1024)
            stream.seek(start)
            data = stream.read(256 * 1024)
            # Avoid splitting a UTF-8 character across requests while a process writes.
            try:
                decoded = data.decode('utf-8')
            except UnicodeDecodeError as exc:
                if exc.reason == 'unexpected end of data' and exc.end == len(data):
                    data = data[:exc.start]
                decoded = data.decode('utf-8', errors='replace')
            return dict(text=decoded, offset=start + len(data), truncated=start > offset)

    def tick(self):
        if time.monotonic() - self.last_probe > 3:
            gpus, error = self.gpu_probe()
            with self.lock:
                self.gpus, self.gpu_error = gpus, error
                self.last_probe = time.monotonic()
        with self.lock, self.db:
            for task in self.rows():
                if task['status'] not in ('starting', 'running', 'canceling'):
                    continue
                directory = self.task_dir(task['id'])
                result_file = directory / 'result.json'
                if result_file.exists():
                    result = json.loads(result_file.read_text())
                    if task['status'] == 'canceling':
                        result['status'] = 'canceled'
                    self.db.execute('UPDATE tasks SET status=?,exit_code=?,finished_at=?,note=? WHERE id=?',
                                    (result['status'], result['exit_code'], result['finished_at'],
                                     result.get('error', ''), task['id']))
                    if result['status'] == 'failed' and self.settings()['stop_on_failure']:
                        self.db.execute("UPDATE settings SET value='true' WHERE key='paused'")
                elif not same_process(task['pid'], task['process_start']):
                    # Lost supervisor: clean up known command before permitting further work.
                    meta = directory / 'process.json'
                    if meta.exists():
                        process = json.loads(meta.read_text())
                        current_start = process_start(process['pid'])
                        if current_start == process['start'] or (current_start is None and group_alive(process['pid'])):
                            stop_group(process['pid'])
                    self.db.execute("UPDATE tasks SET status='interrupted',finished_at=?,note=? WHERE id=?",
                                    (time.time(), '执行器意外退出；队列已暂停，请检查日志后恢复', task['id']))
                    self.db.execute("UPDATE settings SET value='true' WHERE key='paused'")
            for task_id, child in list(self.children.items()):
                if child.poll() is not None:
                    del self.children[task_id]
            tasks = self.rows()
            if self.settings()['paused'] or any(t['status'] in ('starting', 'running', 'canceling') for t in tasks):
                return
            task = next((t for t in tasks if t['status'] == 'queued'), None)
            if not task:
                return
            selected = []
            if task['gpu'] != 'cpu':
                if self.gpu_error:
                    self._note(task, '等待 GPU：' + self.gpu_error)
                    return
                ready = lambda g: (not g['busy'] and g['used_mb'] <= 512 and g['utilization'] <= 10
                                   and g['total_mb'] - g['used_mb'] >= task['min_free_mb'])
                if task['gpu'] == 'auto':
                    selected = [g for g in self.gpus if ready(g)][:1]
                    if not selected:
                        self._note(task, '等待空闲 GPU（无计算进程、已用显存 ≤512 MiB、利用率 ≤10%）')
                        return
                else:
                    indices = task['gpu'].split(',')
                    lookup = {g['index']: g for g in self.gpus}
                    if any(i not in lookup for i in indices):
                        self._note(task, '等待指定 GPU：当前未检测到 ' + task['gpu'])
                        return
                    selected = [lookup[i] for i in indices]
                    if not all(ready(g) for g in selected):
                        self._note(task, '等待指定 GPU 空闲：' + task['gpu'])
                        return
            directory = self.task_dir(task['id'])
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / 'spec.json', dict(command=task['command'], cwd=task['cwd'],
                env={'CUDA_VISIBLE_DEVICES': ','.join(g['uuid'] for g in selected),
                     'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}))
            self.db.execute("UPDATE tasks SET status='starting',started_at=? WHERE id=?", (time.time(), task['id']))
            self.db.commit()
            with (directory / 'supervisor.log').open('ab') as log:
                child = subprocess.Popen([sys.executable, str(ROOT / 'worker.py'), str(directory)],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                         start_new_session=True, close_fds=True)
            self.children[task['id']] = child
            self.db.execute("UPDATE tasks SET status='running',started_at=?,pid=?,process_start=?,assigned=?,note='' WHERE id=?",
                            (time.time(), child.pid, process_start(child.pid),
                             json.dumps([g['index'] for g in selected]), task['id']))
            self.db.commit()
            atomic_json(directory / 'release.json', {'pid': child.pid, 'start': process_start(child.pid)})

    def _note(self, task, note):
        self.db.execute('UPDATE tasks SET note=? WHERE id=?', (note, task['id']))

    def start(self):
        def loop():
            while not self.closed.is_set():
                try:
                    self.tick()
                    self.scheduler_error = None
                except Exception as exc:
                    self.scheduler_error = str(exc)
                    print(f'Scheduler: {exc}', file=sys.stderr, flush=True)
                self.wakeup.wait(.5)
                self.wakeup.clear()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        # Running supervisors deliberately survive a UI/server restart.
        self.closed.set()
        self.wakeup.set()
        if self.thread:
            self.thread.join()
        self.db.close()
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()


def make_handler(queue, port):
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, value, status=200, content_type='application/json; charset=utf-8'):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self, api=False):
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}', f'[::1]:{port}'}
            host = self.headers.get('Host', '')
            if host not in hosts:
                self.send({'error': '仅允许通过本机地址访问；远程使用 SSH 端口转发'}, 403)
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://{h}' for h in hosts}:
                self.send({'error': '拒绝跨站请求'}, 403)
                return False
            if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.send({'error': '拒绝跨站请求'}, 403)
                return False
            if api and not secrets.compare_digest(self.headers.get('X-Queue-Token', ''), token):
                self.send({'error': '页面会话已过期，请刷新页面'}, 403)
                return False
            return True

        def do_GET(self):
            url = urlsplit(self.path)
            if not self.allowed(url.path.startswith('/api/')):
                return
            try:
                if url.path == '/':
                    page = (ROOT / 'static/index.html').read_text().replace('__QUEUE_TOKEN__', token)
                    self.send(page.encode(), content_type='text/html; charset=utf-8')
                elif url.path in ('/style.css', '/app.js'):
                    self.send((ROOT / 'static' / url.path[1:]).read_bytes(), content_type=(
                        'text/css; charset=utf-8' if url.path.endswith('.css') else 'text/javascript; charset=utf-8'))
                elif url.path == '/api/state':
                    self.send(queue.state())
                elif url.path == '/api/processes':
                    mode = parse_qs(url.query).get('mode',['training'])[0]
                    self.send(queue.external_state(mode))
                elif re.fullmatch(r'/api/processes/[\w-]+/log',url.path):
                    identity = url.path.split('/')[3]
                    offset = int(parse_qs(url.query).get('offset',['0'])[0])
                    self.send(queue.process_monitor.logs(identity,offset,queue.external_scope()))
                elif re.fullmatch(r'/api/tasks/\d+/log', url.path):
                    task_id = int(url.path.split('/')[3])
                    offset = int(parse_qs(url.query).get('offset', ['0'])[0])
                    self.send(queue.log(task_id, offset))
                else:
                    self.send({'error': '页面不存在'}, 404)
            except ValueError as exc:
                self.send({'error': str(exc)}, 400)

        def do_POST(self):
            if not self.allowed(True):
                return
            try:
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('请使用 JSON 提交')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 300000:
                    raise ValueError('请求大小无效')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('请求必须是 JSON 对象')
                if self.path == '/api/tasks':
                    self.send({'id': queue.add(data)}, 201)
                elif self.path == '/api/settings':
                    queue.update_settings(data)
                    self.send({'ok': True})
                elif re.fullmatch(r'/api/processes/[\w-]+/(stop|kill|pause|resume|track|bind-log)',self.path):
                    _,_,_,identity,action = self.path.split('/')
                    excluded = queue.external_scope()
                    if action in ('track','bind-log'):
                        path = data.get('path')
                        if action == 'bind-log' and (not isinstance(path,str) or not path.strip()):
                            raise ValueError('请填写已有日志文件的绝对路径')
                        self.send(queue.process_monitor.bind(identity,path if action=='bind-log' else None,excluded))
                    else:
                        self.send(queue.process_monitor.control(identity,action,excluded))
                elif re.fullmatch(r'/api/tasks/\d+/(cancel|retry|top)', self.path):
                    _, _, _, task_id, action = self.path.split('/')
                    self.send({'id': queue.action(int(task_id), action)})
                else:
                    self.send({'error': '接口不存在'}, 404)
            except (ValueError, TypeError) as exc:
                self.send({'error': str(exc)}, 400)
            except Exception:
                self.send({'error': '操作失败，请查看服务终端'}, 500)
                raise
    return Handler


def main():
    parser = argparse.ArgumentParser(description='本机训练任务队列，浏览器访问 http://127.0.0.1:8765')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', default=str(ROOT / '.data'))
    parser.add_argument('--cwd', default=os.getcwd(), help='新任务的默认工作目录')
    args = parser.parse_args()
    if not Path(args.cwd).is_dir():
        parser.error('默认工作目录不存在')
    queue = Queue(args.data_dir, args.cwd)
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(queue, args.port))
        server.daemon_threads = True
        queue.start()
        print(f'Train Queue: http://127.0.0.1:{args.port}', flush=True)
        print(f'数据目录：{queue.directory}；关闭网页不会中断训练。', flush=True)
        def stop(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\n服务已停止，正在运行的训练继续执行；排队任务将在重启服务后继续。', flush=True)
        finally:
            server.server_close()
    finally:
        queue.close()


if __name__ == '__main__':
    main()
