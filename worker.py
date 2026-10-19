"""One durable, detached supervisor per task. Uses only the standard library."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    os.replace(temporary, path)


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def same_process(pid, start):
    if not (pid and start and process_start(pid) == start):
        return False
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return False


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def group_alive(pgid):
    # Exclude zombies: these cannot execute or hold CUDA allocations.
    for item in Path('/proc').iterdir():
        if item.name.isdigit():
            try:
                fields = (item / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != 'Z':
                    return True
            except (OSError, ValueError, IndexError):
                pass
    return False


def stop_group(pgid, grace=5):
    signal_group(pgid, signal.SIGTERM)
    end = time.monotonic() + grace
    while group_alive(pgid) and time.monotonic() < end:
        time.sleep(.1)
    if group_alive(pgid):
        signal_group(pgid, signal.SIGKILL)


def run(directory):
    directory = Path(directory)
    spec = json.loads((directory / 'spec.json').read_text())
    canceled = False
    proc = None

    def on_signal(*_):
        nonlocal canceled
        canceled = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    with (directory / 'output.log').open('ab', buffering=0) as log:
        try:
            # Do not start a command until its supervisor identity is durably committed.
            # A server crash between spawn and commit must never duplicate training.
            deadline = time.monotonic() + 15
            while not (directory / 'release.json').exists():
                if canceled or (directory / 'cancel').exists():
                    canceled = True
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('服务未确认任务启动；训练没有执行')
                time.sleep(.05)
            if not canceled:
                release = json.loads((directory / 'release.json').read_text())
                if release['pid'] != os.getpid() or release['start'] != process_start(os.getpid()):
                    raise RuntimeError('启动确认与当前执行器不匹配；训练没有执行')
            if (directory / 'cancel').exists():
                canceled = True
            if canceled:
                code = None
            else:
                env = os.environ.copy()
                env.update(spec['env'])
                env['PYTHONUNBUFFERED'] = '1'
                read_fd, write_fd = os.pipe()
                with os.fdopen(read_fd, 'rb') as gate_read, os.fdopen(write_fd, 'wb', buffering=0) as gate_write:
                    # EOF on this pipe prevents an unrecorded child from starting training
                    # if the supervisor crashes between Popen and process.json.
                    proc = subprocess.Popen(
                        ['/bin/bash', '-c',
                         'if IFS= read -r queue_start <&"$1"; then exec /bin/bash -lc "$2"; else exit 125; fi',
                         'train-queue', str(gate_read.fileno()), spec['command']], cwd=spec['cwd'],
                        env=env, stdin=subprocess.DEVNULL, stdout=log, pass_fds=(gate_read.fileno(),),
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    atomic_json(directory / 'process.json', {
                        'pid': proc.pid, 'start': process_start(proc.pid),
                    })
                    gate_write.write(b'go\n')
                while proc.poll() is None:
                    if canceled or (directory / 'cancel').exists():
                        canceled = True
                        stop_group(proc.pid)
                        break
                    time.sleep(.2)
                code = proc.wait()
                # Foreground commands are required. Do not leak their background children.
                if group_alive(proc.pid):
                    log.write(b'\n[Train Queue] Cleaning up remaining command processes.\n')
                    stop_group(proc.pid)
                canceled = canceled or (directory / 'cancel').exists()
            atomic_json(directory / 'result.json', {
                'status': 'canceled' if canceled else ('succeeded' if code == 0 else 'failed'),
                'exit_code': code, 'finished_at': time.time(),
            })
        except BaseException as exc:
            if proc is not None:
                stop_group(proc.pid)
                proc.wait()
            log.write(f'\n[Train Queue] Worker error: {exc}\n'.encode())
            atomic_json(directory / 'result.json', {
                'status': 'failed', 'exit_code': None, 'finished_at': time.time(),
                'error': str(exc),
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    run(parser.parse_args().directory)
