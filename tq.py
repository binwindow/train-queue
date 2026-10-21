#!/usr/bin/env python3
"""A foreground GPU queue for tmux. Python 3.10+, Linux, no dependencies."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import io
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time


class QueueError(Exception):
    pass


class Locked(QueueError):
    pass


class Stopped(BaseException):
    def __init__(self, signum):
        self.signum = signum


def say(message):
    print(message, flush=True)


def default_data_dir():
    base = Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local/state')
    if not base.is_absolute():
        raise QueueError('XDG_STATE_HOME 必须是绝对路径，或使用 --data-dir')
    return base / 'gpu-queue-lite'


def queue_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise argparse.ArgumentTypeError('队列名只能包含字母、数字、连字符和下划线，最长 64 字符')
    return value


def gpu_ids(value):
    if not re.fullmatch(r'\d+(,\d+)*', value):
        raise argparse.ArgumentTypeError('GPU 请使用物理编号，例如 2 或 2,3')
    values = [str(int(item)) for item in value.split(',')]
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError('GPU 编号不能重复')
    return ','.join(values)


def gpu_id(value):
    result = gpu_ids(value)
    if ',' in result:
        raise argparse.ArgumentTypeError('评估卡只支持一个编号')
    return result


def positive_number(value):
    try:
        result = float(value)
    except ValueError:
        result = float('nan')
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('请输入有限的正数')
    return result


def memory_mb(value):
    amount = positive_number(value) * 1024
    if not math.isfinite(amount) or amount > 2**63 - 1:
        raise argparse.ArgumentTypeError('显存门槛超出可存储范围')
    return math.ceil(amount)


def directory(value):
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise QueueError(f'工作目录不存在：{path}')
    return str(path)


@contextmanager
def file_locks(folder, names):
    streams = []
    try:
        for name in sorted(set(names)):
            stream = (folder / (name + '.lock')).open('a')
            streams.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise Locked(f'锁正被占用：{name}') from None
        yield
    finally:
        for stream in reversed(streams):
            stream.close()


class Store:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.chmod(0o700)
        self.locks = self.path / 'locks'
        self.locks.mkdir(exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.path / 'queue.sqlite3', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS queues (
                name TEXT PRIMARY KEY, gpus TEXT NOT NULL, monitor_gpu TEXT NOT NULL,
                cwd TEXT NOT NULL, share INTEGER NOT NULL,
                train_free_mb INTEGER, monitor_free_mb INTEGER,
                blocked INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, queue TEXT NOT NULL,
                command TEXT NOT NULL, cwd TEXT NOT NULL,
                train_free_mb INTEGER, monitor_free_mb INTEGER,
                status TEXT NOT NULL DEFAULT 'queued', created_at REAL NOT NULL,
                started_at REAL, finished_at REAL, exit_code INTEGER,
                pid INTEGER, start_ticks TEXT, boot_id TEXT, note TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS task_order ON tasks(queue,status,id);
            CREATE UNIQUE INDEX IF NOT EXISTS one_running ON tasks(queue) WHERE status='running';
        ''')

    def close(self):
        self.db.close()

    def queue(self, name):
        row = self.db.execute('SELECT * FROM queues WHERE name=?', (name,)).fetchone()
        if row is None:
            raise QueueError(f'队列 {name} 尚未配置，请先运行 tq work')
        return dict(row)

    def configure(self, args):
        old = self.db.execute('SELECT * FROM queues WHERE name=?', (args.queue,)).fetchone()
        config = dict(old) if old else dict(name=args.queue, gpus=None, monitor_gpu=None,
                                           cwd=None, share=False, train_free_mb=None,
                                           monitor_free_mb=None, blocked=0)
        for field in ('gpus', 'monitor_gpu', 'share', 'train_free_mb', 'monitor_free_mb'):
            value = getattr(args, field)
            if value is not None:
                config[field] = value
        config['cwd'] = directory(args.cwd or config['cwd'] or os.getcwd())
        if config['gpus'] is None or config['monitor_gpu'] is None:
            raise QueueError('首次启动需要同时指定 --gpus 和 --monitor-gpu')
        if config['monitor_gpu'] in config['gpus'].split(','):
            raise QueueError('评估卡不能与训练卡重叠')
        if config['share'] and (config['train_free_mb'] is None or config['monitor_free_mb'] is None):
            raise QueueError('共享模式需要 --train-free-gib 和 --monitor-free-gib（包含显存余量）')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if old and any(config[key] != old[key] for key in ('gpus', 'monitor_gpu')):
                pending = self.db.execute("SELECT 1 FROM tasks WHERE queue=? AND status IN ('queued','running')",
                                          (args.queue,)).fetchone()
                if pending:
                    raise QueueError('队列还有待运行或运行任务，不能更改卡组；请使用新队列名')
            self.db.execute('''INSERT INTO queues VALUES
                (:name,:gpus,:monitor_gpu,:cwd,:share,:train_free_mb,:monitor_free_mb,:blocked)
                ON CONFLICT(name) DO UPDATE SET gpus=excluded.gpus,monitor_gpu=excluded.monitor_gpu,
                cwd=excluded.cwd,share=excluded.share,train_free_mb=excluded.train_free_mb,
                monitor_free_mb=excluded.monitor_free_mb''', config)
        return self.queue(args.queue)

    def add(self, args):
        if not args.command.strip() or '\0' in args.command:
            raise QueueError('命令不能为空或包含 NUL 字符')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            config = self.queue(args.queue)
            if not config['share'] and (args.train_free_mb or args.monitor_free_mb):
                raise QueueError('显存门槛覆盖仅用于共享模式，请先以 work --share 配置该组')
            cur = self.db.execute('''INSERT INTO tasks
                (queue,command,cwd,train_free_mb,monitor_free_mb,created_at) VALUES (?,?,?,?,?,?)''',
                (args.queue, args.command, directory(args.cwd or config['cwd']),
                 args.train_free_mb, args.monitor_free_mb, time.time()))
            return cur.lastrowid

    def head(self, name):
        row = self.db.execute("SELECT * FROM tasks WHERE queue=? AND status='queued' ORDER BY id LIMIT 1",
                              (name,)).fetchone()
        return dict(row) if row else None

    def claim(self, task_id):
        with self.db:
            return self.db.execute("UPDATE tasks SET status='running',started_at=? WHERE id=? AND status='queued'",
                                   (time.time(), task_id)).rowcount == 1

    def finish(self, task_id, status, code=None, note=''):
        with self.db:
            self.db.execute('UPDATE tasks SET status=?,exit_code=?,finished_at=?,note=? WHERE id=?',
                            (status, code, time.time(), note, task_id))
            if status == 'interrupted':
                self.db.execute('UPDATE queues SET blocked=1 WHERE name=(SELECT queue FROM tasks WHERE id=?)',
                                (task_id,))

    def recover(self, name, resume):
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            stale = self.db.execute("SELECT id FROM tasks WHERE queue=? AND status='running'", (name,)).fetchall()
            if stale:
                self.db.execute("UPDATE tasks SET status='interrupted',finished_at=?,note=? WHERE queue=? AND status='running'",
                                (time.time(), '执行器异常退出；训练可能仍在运行，请人工检查', name))
                self.db.execute('UPDATE queues SET blocked=1 WHERE name=?', (name,))
        if not self.queue(name)['blocked']:
            return
        if not resume:
            raise QueueError('该组有中断任务。先用 tq list 查看并检查训练进程，再用 tq work 队列名 --resume 恢复；不会自动重跑')
        interrupted = self.db.execute("SELECT * FROM tasks WHERE queue=? AND status='interrupted'", (name,)).fetchall()
        for task in interrupted:
            if process_matches(task):
                raise QueueError(f"任务 #{task['id']} 的命令进程 PID {task['pid']} 仍在运行，请先处理它再恢复")
        with self.db:
            self.db.execute('UPDATE queues SET blocked=0 WHERE name=?', (name,))


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return fields[19], boot, fields[0]
    except (OSError, IndexError):
        return None, None, None


def process_matches(task):
    if not task['pid']:
        return False
    start, boot, state = process_identity(task['pid'])
    return bool(start and start == task['start_ticks'] and boot == task['boot_id'] and state not in ('Z', 'X'))


def probe_gpus(include_processes=True):
    def query(fields):
        try:
            result = subprocess.run(['nvidia-smi', fields, '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, check=True, timeout=5)
            return list(csv.reader(io.StringIO(result.stdout)))
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, 'stderr', None) or str(exc)
            raise QueueError('GPU 检测失败：' + detail.strip()[:250]) from None

    rows = query('--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu')
    try:
        gpus = {}
        for row in rows:
            index, uuid, used, free, utilization = [cell.strip() for cell in row]
            if not index.isdigit() or not re.fullmatch(r'GPU-[A-Za-z0-9-]+', uuid):
                raise ValueError()
            gpus[str(int(index))] = dict(uuid=uuid, used=int(used), free=int(free),
                                         utilization=int(utilization) if utilization.isdigit() else None,
                                         busy=False)
    except (ValueError, TypeError):
        raise QueueError('GPU 检测结果无效，未启动任务') from None
    if not gpus:
        raise QueueError('未检测到 NVIDIA GPU')
    if include_processes:
        busy = set()
        for row in query('--query-compute-apps=gpu_uuid,pid'):
            if len(row) != 2 or not row[0].strip().startswith('GPU-') or not row[1].strip().isdigit():
                raise QueueError('GPU 计算进程检测结果无效，未启动任务')
            busy.add(row[0].strip())
        for gpu in gpus.values():
            gpu['busy'] = gpu['uuid'] in busy
    return gpus


def readiness(config, task, gpus):
    roles = [(index, 'train') for index in config['gpus'].split(',')]
    roles.append((config['monitor_gpu'], 'monitor'))
    for index, role in roles:
        gpu = gpus.get(index)
        if gpu is None:
            return f'未检测到 GPU {index}'
        if config['share']:
            key = role + '_free_mb'
            threshold = task[key] if task[key] is not None else config[key]
            if gpu['free'] < threshold:
                return f"GPU {index} 空闲 {gpu['free']/1024:.2f} GiB，需要 {threshold/1024:.2f} GiB"
        elif gpu['busy'] or gpu['used'] > 512 or gpu['utilization'] is None or gpu['utilization'] > 10:
            return f"GPU {index} 未空闲（计算进程={gpu['busy']}，已用={gpu['used']} MiB，利用率={gpu['utilization']}%）"
    return None


def interrupt_and_wait(proc, forward=True):
    """Relay Ctrl+C only within our own foreground job, never the caller's group."""
    if forward and proc.poll() is None:
        try:
            if os.getpgrp() == os.getpid():
                previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    os.killpg(os.getpgrp(), signal.SIGINT)
                finally:
                    signal.signal(signal.SIGINT, previous)
            else:
                proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
    while True:
        try:
            return proc.wait()
        except Stopped:
            say('仍在等待原训练命令收尾；不自动强杀，请在另一终端检查进程。')


def execute(store, config, task):
    env = os.environ.copy()
    env.update(GPU_IDS=config['gpus'], MONITOR_GPU_ID=config['monitor_gpu'])
    say(f"\n>>> #{task['id']} [{config['name']}] cwd={task['cwd']}\n{task['command']}\n")
    try:
        # Keep the controlling terminal, standard streams and original process group.
        proc = subprocess.Popen(['/bin/bash', '-c', task['command']], cwd=task['cwd'], env=env)
    except (OSError, ValueError) as exc:
        store.finish(task['id'], 'failed', note=str(exc))
        say(f"<<< #{task['id']} 启动失败：{exc}")
        return
    try:
        start, boot, _ = process_identity(proc.pid)
        with store.db:
            store.db.execute('UPDATE tasks SET pid=?,start_ticks=?,boot_id=? WHERE id=?',
                             (proc.pid, start, boot, task['id']))
        code = proc.wait()
    except Stopped as exc:
        say('\n收到中断，等待原训练命令收尾…')
        # A terminal sends SIGINT to the entire foreground group already. A
        # second SIGINT could interrupt the supervisor's finally/cleanup block.
        code = interrupt_and_wait(proc, forward=exc.signum != signal.SIGINT)
        store.finish(task['id'], 'interrupted', code, '用户中断；检查进程后使用 work --resume 恢复')
        raise
    except BaseException:
        interrupt_and_wait(proc)
        raise
    if code in (-signal.SIGINT, -signal.SIGTERM, 130, 143):
        store.finish(task['id'], 'interrupted', code, '训练命令被中断；检查进程后使用 work --resume 恢复')
        raise Stopped(signal.SIGINT)
    store.finish(task['id'], 'succeeded' if code == 0 else 'failed', code)
    say(f"\n<<< #{task['id']} 退出码 {code}；继续等待下一项")


def work(store, args):
    with file_locks(store.locks, ['queue-' + args.queue]):
        config = store.configure(args)
        store.recover(args.queue, args.resume)
        say(f"队列 {args.queue} | 训练卡 {config['gpus']} | 评估卡 {config['monitor_gpu']} | "
            f"{'共享显存模式' if config['share'] else '等待空闲模式'}\n"
            f"cwd={config['cwd']}\nPython={sys.executable}\n数据目录={store.path}")
        last_message, samples, last_task = None, 0, None
        while True:
            task = store.head(args.queue)
            message = '队列为空，等待从其他终端添加任务…'
            if task:
                if task['id'] != last_task:
                    samples = 0
                    last_task = task['id']
                try:
                    gpus = probe_gpus(include_processes=not config['share'])
                    reason = readiness(config, task, gpus)
                    if reason:
                        samples = 0
                        message = f"#{task['id']} 等待：{reason}"
                    else:
                        samples += 1
                        message = f"#{task['id']} 检测满足，等待第二次确认…"
                        if samples >= 2:
                            indices = config['gpus'].split(',') + [config['monitor_gpu']]
                            uuids = {i: gpus[i]['uuid'] for i in indices}
                            with file_locks(store.locks, ['gpu-' + uuid for uuid in uuids.values()]):
                                fresh = probe_gpus(include_processes=not config['share'])
                                reason = readiness(config, task, fresh)
                                if reason:
                                    message = f"#{task['id']} 等待：{reason}"
                                elif any(fresh[i]['uuid'] != uuid for i, uuid in uuids.items()):
                                    message = 'GPU 编号映射已变化，重新检查…'
                                elif store.claim(task['id']):
                                    execute(store, config, task)
                                    message = None
                            samples = 0
                except Locked:
                    message = f"#{task['id']} 等待：相关 GPU 正被另一个队列使用"
                    samples = 0
                except QueueError as exc:
                    message = f"#{task['id']} 等待：{exc}"
                    samples = 0
            else:
                samples = 0
            if message and message != last_message:
                say(message)
            last_message = message
            time.sleep(args.poll_seconds)


def list_tasks(store, args):
    configs = store.db.execute('SELECT * FROM queues WHERE name=?' if args.queue else 'SELECT * FROM queues ORDER BY name',
                               (args.queue,) if args.queue else ()).fetchall()
    if args.queue and not configs:
        raise QueueError(f'队列不存在：{args.queue}')
    if not configs:
        say('暂无队列，请先运行 tq work')
    for config in configs:
        say(f"\n[{config['name']}] 训练={config['gpus']} 评估={config['monitor_gpu']} "
            f"模式={'共享' if config['share'] else '空闲'}{' | 需要 --resume' if config['blocked'] else ''}")
        say('  cwd=' + config['cwd'])
        if config['share']:
            say(f"  空闲显存门槛：每张训练卡 {config['train_free_mb']/1024:g} GiB，评估卡 {config['monitor_free_mb']/1024:g} GiB")
        rows = store.db.execute('SELECT * FROM tasks WHERE queue=? ORDER BY id', (config['name'],)).fetchall()
        if not rows:
            say('  暂无任务')
        for row in rows:
            code = '' if row['exit_code'] is None else f" exit={row['exit_code']}"
            pid = f" pid={row['pid']}" if row['status'] in ('running', 'interrupted') and row['pid'] else ''
            say(f"  #{row['id']} {row['status']}{code}{pid} | {row['command']}")
            if row['cwd'] != config['cwd']:
                say('    cwd=' + row['cwd'])
            for role, label in (('train', '每张训练卡'), ('monitor', '评估卡')):
                threshold = row[role + '_free_mb']
                if threshold is not None:
                    say(f'    覆盖门槛：{label} {threshold/1024:g} GiB')
            if row['note']:
                say('    ' + row['note'])


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument('--data-dir', help='数据目录；默认 $XDG_STATE_HOME/gpu-queue-lite 或 ~/.local/state/gpu-queue-lite')
    commands = root.add_subparsers(dest='action', required=True)
    run = commands.add_parser('work', help='在当前 tmux 窗口执行一组队列')
    run.add_argument('queue', type=queue_name)
    run.add_argument('--gpus', type=gpu_ids)
    run.add_argument('--monitor-gpu', type=gpu_id)
    run.add_argument('--share', action=argparse.BooleanOptionalAction, default=None)
    run.add_argument('--cwd', help='首次默认当前目录，重启默认该组原目录')
    run.add_argument('--resume', action='store_true', help='检查中断任务后明确恢复，不重跑中断项')
    run.add_argument('--poll-seconds', type=positive_number, default=2)
    add = commands.add_parser('add', help='追加一条完整 Bash 命令')
    add.add_argument('queue', type=queue_name)
    add.add_argument('--command', '-c', required=True)
    add.add_argument('--cwd', help='本任务目录；默认该组目录')
    for command in (run, add):
        command.add_argument('--train-free-gib', dest='train_free_mb', type=memory_mb,
                             metavar='GiB', help='共享模式下每张训练卡的空闲显存门槛，包含余量')
        command.add_argument('--monitor-free-gib', dest='monitor_free_mb', type=memory_mb,
                             metavar='GiB', help='共享模式下评估卡的空闲显存门槛，包含余量')
    listing = commands.add_parser('list', help='查看队列配置和任务')
    listing.add_argument('queue', type=queue_name, nargs='?')
    cancel = commands.add_parser('cancel', help='取消尚未启动的任务')
    cancel.add_argument('id', type=int)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    store = None
    def stop(signum, _frame):
        raise Stopped(signum)
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        store = Store(args.data_dir or default_data_dir())
        if args.action == 'work':
            work(store, args)
        elif args.action == 'add':
            say(f'#{store.add(args)} 已加入 {args.queue}')
        elif args.action == 'list':
            list_tasks(store, args)
        elif args.action == 'cancel':
            with store.db:
                count = store.db.execute("UPDATE tasks SET status='canceled',finished_at=? WHERE id=? AND status='queued'",
                                         (time.time(), args.id)).rowcount
                if not count:
                    raise QueueError('任务不存在或已启动；只允许取消待运行任务')
            say(f'#{args.id} 已取消')
        return 0
    except Stopped as exc:
        say('\n执行器已停止，后续任务保留。')
        return 128 + exc.signum
    except (QueueError, OSError, sqlite3.Error) as exc:
        print(f'错误：{exc}', file=sys.stderr, flush=True)
        return 1
    finally:
        if store:
            store.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
