import argparse
import concurrent.futures
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tq

ROOT = Path(tq.__file__).parent


def gpu(index, free=24000, used=0, busy=False, utilization=0):
    return dict(uuid=f'GPU-test-{index}', free=free, used=used, busy=busy,
                utilization=utilization)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'state'
        self.cwd = self.root / 'project with spaces'
        self.cwd.mkdir()
        self.store = tq.Store(self.data)
        self.workers = []
        self.handles = []
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.gpu_file = self.root / 'gpus.csv'
        self.apps_file = self.root / 'apps.csv'
        self.write_gpus()
        self.apps_file.write_text('')
        probe = self.bin / 'nvidia-smi'
        probe.write_text('#!/bin/sh\ncase "$1" in\n'
                         '*query-compute-apps*) cat "$TQ_TEST_APPS" ;;\n'
                         '*query-gpu*) cat "$TQ_TEST_GPUS" ;;\n'
                         '*) exit 1 ;;\nesac\n')
        probe.chmod(0o755)
        self.env = {**os.environ, 'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
                    'TQ_TEST_GPUS': str(self.gpu_file), 'TQ_TEST_APPS': str(self.apps_file),
                    'CUDA_VISIBLE_DEVICES': 'unchanged-parent-mask', 'TQ_TEST_VALUE': 'from-worker'}

    def tearDown(self):
        for proc in self.workers:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    pass
            # These groups contain only processes created by this test.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=3)
        for handle in self.handles:
            handle.close()
        self.store.close()
        self.tmp.cleanup()

    def write_gpus(self, free=24000, used=0, utilization=0):
        temporary = self.gpu_file.with_suffix('.tmp')
        temporary.write_text(''.join(f'{i}, GPU-test-{i}, {used}, {free}, {utilization}\n' for i in range(6)))
        temporary.replace(self.gpu_file)

    def configure(self, name='pair-a', *options):
        args = tq.parser().parse_args(['work', name, '--gpus', '2,3', '--monitor-gpu', '0',
                                      '--cwd', str(self.cwd), *options])
        return self.store.configure(args)

    def add(self, command, name='pair-a', *options):
        args = tq.parser().parse_args(['add', name, '--command', command, *options])
        return self.store.add(args)

    def task(self, task_id):
        return dict(self.store.db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone())

    def cli(self, *arguments, success=True, env=None):
        result = subprocess.run([sys.executable, str(ROOT / 'tq.py'), '--data-dir', str(self.data), *arguments],
                                capture_output=True, text=True, env=env or self.env, timeout=15)
        self.assertEqual(result.returncode, 0 if success else 1, result.stdout + result.stderr)
        return result

    def worker(self, name='pair-a', *options):
        log = (self.root / f'worker-{len(self.workers)}.log').open('w+')
        self.handles.append(log)
        proc = subprocess.Popen([sys.executable, str(ROOT / 'tq.py'), '--data-dir', str(self.data),
                                 'work', name, '--poll-seconds', '0.04', *options],
                                env=self.env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        self.workers.append(proc)
        return proc, log

    def wait_for(self, condition, timeout=8):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if condition():
                return
            time.sleep(0.025)
        logs = []
        for handle in self.handles:
            handle.flush()
            handle.seek(0)
            logs.append(handle.read())
        self.fail('Timed out\n' + '\n'.join(logs))

    def test_defaults_and_private_state(self):
        with patch.dict(os.environ, {'XDG_STATE_HOME': str(self.root / 'xdg')}):
            self.assertEqual(tq.default_data_dir(), self.root / 'xdg/gpu-queue-lite')
        with patch.dict(os.environ, {'XDG_STATE_HOME': ''}):
            self.assertEqual(tq.default_data_dir(), Path.home() / '.local/state/gpu-queue-lite')
        with patch.dict(os.environ, {'XDG_STATE_HOME': 'relative'}):
            with self.assertRaises(tq.QueueError):
                tq.default_data_dir()
        self.assertEqual(stat.S_IMODE(self.data.stat().st_mode), 0o700)

    def test_configuration_validation(self):
        for options in (['--gpus', '0,2'], ['--share'], ['--share', '--train-free-gib', '18']):
            with self.assertRaises(tq.QueueError):
                self.configure('pair-a', *options)
        with self.assertRaises(tq.QueueError):
            self.store.configure(tq.parser().parse_args(['work', 'unconfigured']))
        for value in ('2,2', '-1', 'GPU-123', '2 3'):
            with self.assertRaises(argparse.ArgumentTypeError):
                tq.gpu_ids(value)
        for value in ('0', '-1', 'nan', 'inf', '1e308'):
            with self.assertRaises(argparse.ArgumentTypeError):
                tq.memory_mb(value)
        self.assertEqual(tq.gpu_ids('02,3'), '2,3')

    def test_card_configuration_cannot_change_under_pending_tasks(self):
        self.configure()
        self.add('true')
        with self.assertRaises(tq.QueueError):
            self.configure('pair-a', '--gpus', '4,5')
        self.assertEqual(self.store.queue('pair-a')['gpus'], '2,3')

    def test_dynamic_fifo_failure_continues_and_live_output(self):
        self.configure()
        first = self.add("printf 'live-output\\n'; sleep 0.4; printf 'first\\n' >> order")
        proc, log = self.worker()
        self.wait_for(lambda: self.task(first)['status'] == 'running')
        second = self.add("printf 'second\\n' >> order; exit 7")
        third = self.add("printf 'third\\n' >> order")
        self.wait_for(lambda: self.task(third)['status'] == 'succeeded')
        self.assertEqual((self.cwd / 'order').read_text(), 'first\nsecond\nthird\n')
        self.assertEqual(self.task(second)['status'], 'failed')
        self.assertEqual(self.task(second)['exit_code'], 7)
        for a, b in ((first, second), (second, third)):
            self.assertLessEqual(self.task(a)['finished_at'], self.task(b)['started_at'])
        log.seek(0)
        self.assertIn('live-output', log.read())
        self.assertIsNone(proc.poll())

    def test_environment_cwd_multiline_and_exact_command(self):
        self.configure()
        command = shlex.join([sys.executable, '-c',
                             'import os,json; print("中文",flush=True); '
                             'open("environment.json","w").write(json.dumps({k:os.environ.get(k) '
                             'for k in ["GPU_IDS","MONITOR_GPU_ID","CUDA_VISIBLE_DEVICES","TQ_TEST_VALUE"]}))'])
        command = "printf '%s\\n' 'literal $HOME; 中文' > literal\n" + command
        self.cli('add', 'pair-a', '--command', command)
        task = self.store.head('pair-a')
        self.assertEqual(task['command'], command)
        self.worker()
        self.wait_for(lambda: self.task(task['id'])['status'] == 'succeeded')
        data = json.loads((self.cwd / 'environment.json').read_text())
        self.assertEqual(data, dict(GPU_IDS='2,3', MONITOR_GPU_ID='0',
                                    CUDA_VISIBLE_DEVICES='unchanged-parent-mask', TQ_TEST_VALUE='from-worker'))
        self.assertEqual((self.cwd / 'literal').read_text(), 'literal $HOME; 中文\n')

    def test_other_group_can_run_in_parallel(self):
        self.configure('a')
        self.configure('b', '--gpus', '4,5', '--monitor-gpu', '1')
        a = self.add('sleep 1', 'a')
        b = self.add('sleep 1', 'b')
        self.worker('a')
        self.worker('b')
        self.wait_for(lambda: self.task(a)['status'] == self.task(b)['status'] == 'running')
        self.wait_for(lambda: self.task(a)['status'] == self.task(b)['status'] == 'succeeded')

    def test_default_waits_for_compute_process_even_with_free_memory(self):
        self.configure()
        self.apps_file.write_text('GPU-test-2, 999999\n')
        task = self.add('touch started')
        self.worker()
        time.sleep(0.35)
        self.assertEqual(self.task(task)['status'], 'queued')
        self.assertFalse((self.cwd / 'started').exists())
        self.apps_file.write_text('')
        self.wait_for(lambda: self.task(task)['status'] == 'succeeded')

    def test_shared_mode_allows_busy_card_with_enough_free_memory(self):
        self.configure('pair-a', '--share', '--train-free-gib', '18', '--monitor-free-gib', '6')
        self.write_gpus(free=21000, used=3000, utilization=95)
        self.apps_file.write_text('GPU-test-2, 999999\n')
        task = self.add('true')
        self.worker()
        self.wait_for(lambda: self.task(task)['status'] == 'succeeded')

    def test_shared_mode_checks_each_card_monitor_and_task_override(self):
        config = self.configure('pair-a', '--share', '--train-free-gib', '18', '--monitor-free-gib', '6')
        task_id = self.add('true', 'pair-a', '--train-free-gib', '20', '--monitor-free-gib', '8')
        task = self.task(task_id)
        gpus = {str(i): gpu(i) for i in range(4)}
        self.assertIsNone(tq.readiness(config, task, gpus))
        gpus['3']['free'] = 19000
        self.assertIn('GPU 3', tq.readiness(config, task, gpus))
        gpus['3']['free'] = 24000
        gpus['0']['free'] = 7000
        self.assertIn('GPU 0', tq.readiness(config, task, gpus))
        del gpus['0']
        self.assertIn('GPU 0', tq.readiness(config, task, gpus))

    def test_insufficient_memory_and_probe_failure_do_not_start(self):
        self.configure('pair-a', '--share', '--train-free-gib', '18', '--monitor-free-gib', '6')
        self.write_gpus(free=10000, used=14000)
        task = self.add('touch started')
        self.worker()
        time.sleep(0.3)
        self.assertEqual(self.task(task)['status'], 'queued')
        self.gpu_file.write_text('invalid probe output\n')
        time.sleep(0.2)
        self.assertEqual(self.task(task)['status'], 'queued')
        self.assertFalse((self.cwd / 'started').exists())
        self.write_gpus()
        self.wait_for(lambda: self.task(task)['status'] == 'succeeded')

    def test_requires_two_samples_and_rechecks_after_lock(self):
        self.configure()
        task = self.add('touch started')
        calls = []
        def probe(**_):
            calls.append(None)
            if len(calls) == 3:
                raise tq.Stopped(signal.SIGINT)
            return {str(i): gpu(i) for i in range(4)}
        args = tq.parser().parse_args(['work', 'pair-a', '--poll-seconds', '0.01'])
        with patch('tq.probe_gpus', side_effect=probe), redirect_stdout(io.StringIO()):
            with self.assertRaises(tq.Stopped):
                tq.work(self.store, args)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.task(task)['status'], 'queued')

    def test_gpu_lock_and_duplicate_worker_prevent_double_execution(self):
        self.configure()
        task = self.add('true')
        with tq.file_locks(self.store.locks, ['gpu-GPU-test-2']):
            self.worker()
            time.sleep(0.35)
            self.assertEqual(self.task(task)['status'], 'queued')
            self.cli('work', 'pair-a', success=False)
        self.wait_for(lambda: self.task(task)['status'] == 'succeeded')

    def test_partial_lock_acquisition_is_released(self):
        with tq.file_locks(self.store.locks, ['b']):
            with self.assertRaises(tq.Locked):
                with tq.file_locks(self.store.locks, ['a', 'b']):
                    self.fail('Should not acquire b twice')
            with tq.file_locks(self.store.locks, ['a']):
                pass

    def test_concurrent_cli_additions_and_cancel(self):
        self.configure()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda i: self.cli('add', 'pair-a', '-c', f'echo {i}'), range(16)))
        self.assertEqual(len(results), 16)
        rows = self.store.db.execute('SELECT * FROM tasks ORDER BY id').fetchall()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({row['command'] for row in rows}), 16)
        task_id = rows[0]['id']
        self.cli('cancel', str(task_id))
        self.assertEqual(self.task(task_id)['status'], 'canceled')
        self.cli('cancel', str(task_id), success=False)
        self.assertIn('canceled', self.cli('list', 'pair-a').stdout)

    def test_abandoned_task_blocks_until_explicit_resume_without_retry(self):
        self.configure()
        abandoned = self.add('touch must-not-run')
        pending = self.add('touch next')
        self.store.claim(abandoned)
        self.cli('work', 'pair-a', success=False)
        self.assertEqual(self.task(abandoned)['status'], 'interrupted')
        self.assertEqual(self.task(pending)['status'], 'queued')
        self.worker('pair-a', '--resume')
        self.wait_for(lambda: self.task(pending)['status'] == 'succeeded')
        self.assertFalse((self.cwd / 'must-not-run').exists())

    def test_resume_rejects_still_running_command_and_boot_identity(self):
        self.configure()
        task_id = self.add('true')
        self.store.claim(task_id)
        start, boot, _ = tq.process_identity(os.getpid())
        with self.store.db:
            self.store.db.execute('UPDATE tasks SET pid=?,start_ticks=?,boot_id=? WHERE id=?',
                                  (os.getpid(), start, boot, task_id))
        with self.assertRaises(tq.QueueError):
            self.store.recover('pair-a', resume=True)
        row = self.task(task_id)
        self.assertTrue(tq.process_matches(row))
        self.assertFalse(tq.process_matches({**row, 'boot_id': 'previous-boot'}))

    def test_killed_worker_does_not_start_next_or_duplicate_old_command(self):
        self.configure()
        first = self.add('echo start >> launches; touch ready; sleep 60')
        second = self.add('touch next')
        proc, _ = self.worker()
        self.wait_for(lambda: (self.cwd / 'ready').exists() and self.task(first)['pid'] is not None)
        proc.kill()
        proc.wait(timeout=3)
        self.cli('work', 'pair-a', success=False)
        self.cli('work', 'pair-a', '--resume', success=False)
        self.assertEqual(self.task(second)['status'], 'queued')
        os.killpg(proc.pid, signal.SIGKILL)
        self.wait_for(lambda: not tq.process_matches(self.task(first)))
        self.worker('pair-a', '--resume')
        self.wait_for(lambda: self.task(second)['status'] == 'succeeded')
        self.assertEqual((self.cwd / 'launches').read_text(), 'start\n')

    def test_launch_failure_continues_to_next_task(self):
        self.configure()
        vanished = self.cwd / 'vanished'
        vanished.mkdir()
        first = self.add('true', 'pair-a', '--cwd', str(vanished))
        vanished.rmdir()
        second = self.add('true')
        self.worker()
        self.wait_for(lambda: self.task(second)['status'] == 'succeeded')
        self.assertEqual(self.task(first)['status'], 'failed')
        self.assertIn('vanished', self.task(first)['note'])

    def test_ctrl_c_runs_supervisor_cleanup_and_preserves_pending(self):
        self.configure()
        script = self.cwd / 'supervisor.py'
        script.write_text('import subprocess, sys, time\n'
                          'from pathlib import Path\n'
                          'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)\n'
                          'Path("ready").write_text(str(child.pid))\n'
                          'try:\n'
                          '    time.sleep(60)\n'
                          'except KeyboardInterrupt:\n'
                          '    pass\n'
                          'finally:\n'
                          '    time.sleep(0.2)\n'
                          '    child.terminate()\n'
                          '    child.wait()\n'
                          '    Path("cleaned").touch()\n')
        task_id = self.add('exec ' + shlex.join([sys.executable, str(script)]))
        pending = self.add('touch must-not-run')
        proc, _ = self.worker()
        self.wait_for(lambda: (self.cwd / 'ready').exists())
        os.killpg(proc.pid, signal.SIGINT)
        self.assertEqual(proc.wait(timeout=6), 130)
        self.assertTrue((self.cwd / 'cleaned').exists())
        self.assertEqual(self.task(task_id)['status'], 'interrupted')
        self.assertEqual(self.task(pending)['status'], 'queued')
        self.assertFalse((self.cwd / 'must-not-run').exists())

    def test_waiting_interrupt_preserves_queued_tasks(self):
        self.configure()
        self.apps_file.write_text('GPU-test-2, 999999\n')
        task_id = self.add('true')
        proc, _ = self.worker()
        time.sleep(0.25)
        proc.send_signal(signal.SIGINT)
        self.assertEqual(proc.wait(timeout=4), 130)
        self.assertEqual(self.task(task_id)['status'], 'queued')
        self.assertFalse(self.store.queue('pair-a')['blocked'])

    def test_probe_shared_skips_process_query_and_invalid_data_waits(self):
        response = subprocess.CompletedProcess([], 0, '2, GPU-example-2, 3000, 21000, N/A\n', '')
        with patch('tq.subprocess.run', return_value=response) as run:
            result = tq.probe_gpus(include_processes=False)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result['2']['free'], 21000)
        with patch('tq.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'malformed', '')):
            with self.assertRaises(tq.QueueError):
                tq.probe_gpus()


if __name__ == '__main__':
    unittest.main()
