import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import Queue, make_handler, normalize_submission
from worker import atomic_json, group_alive, same_process, process_start


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.queue = Queue(self.root / 'data', self.root, gpu_probe=lambda: ([], 'no driver'))

    def tearDown(self):
        # Only cancel processes owned by this test's isolated task directory.
        for task in self.queue.state()['tasks']:
            if task['status'] in ('starting', 'running', 'canceling'):
                self.queue.action(task['id'], 'cancel')
        self.wait_for(lambda: not any(t['status'] in ('starting', 'running', 'canceling') for t in self.queue.state()['tasks']))
        self.queue.close()
        self.tmp.cleanup()

    def add(self, command, **extra):
        return self.queue.add(dict(command=command, gpu='cpu', **extra))

    def task(self, task_id):
        return next(t for t in self.queue.state()['tasks'] if t['id'] == task_id)

    def wait_for(self, condition, timeout=12):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.queue.tick()
            if condition():
                return
            time.sleep(.05)
        self.fail('condition timed out: ' + json.dumps(self.queue.state()))

    def test_fifo_real_processes_and_failure_continues(self):
        first = self.add("printf 'first-start\\n'; sleep 0.3; printf 'first-end\\n'")
        second = self.add("printf 'second\\n'; exit 7")
        third = self.add("printf 'third\\n'")
        self.wait_for(lambda: self.task(third)['status'] == 'succeeded')
        a, b, c = [self.task(i) for i in (first, second, third)]
        self.assertLessEqual(a['finished_at'], b['started_at'])
        self.assertLessEqual(b['finished_at'], c['started_at'])
        self.assertEqual(b['exit_code'], 7)
        self.assertEqual(b['status'], 'failed')
        self.assertIn('first-end', self.queue.log(first)['text'])

    def test_failure_pauses_persistently(self):
        self.queue.update_settings({'stop_on_failure': True})
        first, second = self.add('exit 4'), self.add('echo later')
        self.wait_for(lambda: self.task(first)['status'] == 'failed')
        self.assertTrue(self.queue.settings()['paused'])
        self.assertEqual(self.task(second)['status'], 'queued')
        self.queue.update_settings({'paused': False})
        self.wait_for(lambda: self.task(second)['status'] == 'succeeded')

    def test_no_gpu_never_launches_gpu_task_and_fifo_blocks_cpu(self):
        gpu = self.queue.add({'command':'echo gpu','gpu':'auto'})
        cpu = self.add('echo cpu')
        self.queue.tick()
        self.assertEqual(self.task(gpu)['status'], 'queued')
        self.assertIn('等待 GPU', self.task(gpu)['note'])
        self.assertEqual(self.task(cpu)['status'], 'queued')
        self.queue.action(gpu, 'cancel')
        self.wait_for(lambda: self.task(cpu)['status'] == 'succeeded')

    def test_gpu_allocation_busy_and_multi_gpu(self):
        gpus = [dict(index=str(i), uuid=f'GPU-test-{i}',name='test',total_mb=24000,
                     used_mb=0, utilization=0,busy=(i == 1)) for i in range(2)]
        self.queue.gpu_probe = lambda:(gpus,None)
        task = self.queue.add({'command':"printf '%s' \"$CUDA_VISIBLE_DEVICES\"", 'gpu':'0,1'})
        self.queue.tick()
        self.assertEqual(self.task(task)['status'],'queued')
        gpus[1]['busy'] = False
        self.queue.last_probe = 0
        self.wait_for(lambda:self.task(task)['status']=='succeeded')
        self.assertEqual(self.task(task)['assigned'],['0','1'])
        self.assertEqual(self.queue.log(task)['text'],'GPU-test-0,GPU-test-1')

    def test_cancel_kills_command_group(self):
        task = self.add('sleep 60 & wait')
        meta = self.queue.task_dir(task) / 'process.json'
        self.wait_for(meta.exists)
        process = json.loads(meta.read_text())
        self.assertTrue(group_alive(process['pid']))
        self.queue.action(task, 'cancel')
        self.wait_for(lambda: self.task(task)['status'] == 'canceled')
        self.assertFalse(group_alive(process['pid']))

    def test_cancel_escalates_for_term_ignoring_children(self):
        task = self.add("trap '' TERM; sleep 60 & wait")
        meta = self.queue.task_dir(task) / 'process.json'
        self.wait_for(meta.exists)
        time.sleep(.1)
        process = json.loads(meta.read_text())
        self.queue.action(task, 'cancel')
        self.wait_for(lambda: self.task(task)['status'] == 'canceled')
        self.assertFalse(group_alive(process['pid']))

    def test_running_task_survives_server_restart_without_duplicate(self):
        task = self.add("echo start >> marker; sleep 0.6; echo end >> marker")
        self.wait_for(lambda: (self.root / 'marker').exists())
        pid = self.task(task)['pid']
        supervisor = self.queue.children[task]
        self.queue.close()
        self.queue = Queue(self.root / 'data', self.root, gpu_probe=lambda: ([], 'no driver'))
        self.assertEqual(self.task(task)['pid'], pid)
        self.wait_for(lambda: self.task(task)['status'] == 'succeeded')
        self.assertEqual((self.root / 'marker').read_text(), 'start\nend\n')
        supervisor.wait(timeout=3)

    def test_dead_worker_interrupts_and_pauses(self):
        task = self.add('sleep 60')
        meta = self.queue.task_dir(task) / 'process.json'
        self.wait_for(meta.exists)
        process = json.loads(meta.read_text())
        os.kill(self.task(task)['pid'], signal.SIGKILL)
        self.wait_for(lambda: self.task(task)['status'] == 'interrupted')
        self.assertTrue(self.queue.settings()['paused'])
        self.assertFalse(group_alive(process['pid']))

    def test_supervisor_cannot_train_before_durable_release(self):
        directory = self.root / 'unreleased'
        directory.mkdir()
        atomic_json(directory / 'spec.json', {'command':'echo bad > unexpected', 'cwd':str(self.root), 'env':{}})
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[1] / 'worker.py'), str(directory)])
        try:
            time.sleep(.3)
            self.assertFalse((self.root / 'unexpected').exists())
            (directory / 'cancel').touch()
            child.wait(timeout=4)
            self.assertEqual(json.loads((directory / 'result.json').read_text())['status'], 'canceled')
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_lock_prevents_duplicate_scheduler(self):
        with self.assertRaises(RuntimeError):
            Queue(self.root / 'data', self.root)

    def test_pause_top_retry_and_validation(self):
        self.queue.update_settings({'paused':True})
        first, second = self.add('echo first'), self.add('echo second')
        self.queue.action(second,'top')
        self.queue.tick()
        self.assertEqual(self.task(second)['status'],'queued')
        self.queue.update_settings({'paused':False})
        self.wait_for(lambda:self.task(first)['status']=='succeeded')
        self.assertLessEqual(self.task(second)['finished_at'],self.task(first)['started_at'])
        retry = self.queue.action(second,'retry')
        self.assertNotEqual(retry,second)
        self.wait_for(lambda:self.task(retry)['status']=='succeeded')
        for command in ('CUDA_VISIBLE_DEVICES=0 python train.py','export CUDA_VISIBLE_DEVICES=0\npython train.py'):
            normalized = normalize_submission({'command':command},self.root)
            self.assertEqual(normalized[0],'python train.py')
            self.assertEqual(normalized[2],'0')
        for data in ({'command':'echo hi','cwd':'/nonexistent'}, {'command':'echo hi','gpu':'0,0'},
                     {'command':'echo hi','gpu':'bad'}, {'command':'export CUDA_VISIBLE_DEVICES=0; python train.py','gpu':'1'}):
            with self.assertRaises(ValueError):
                self.queue.add(data)

    def test_log_utf8_chunk_boundary_and_tail(self):
        task = self.add('echo hi')
        path = self.queue.task_dir(task)
        path.mkdir(parents=True)
        (path / 'output.log').write_bytes('中文'.encode()[:-1])
        first = self.queue.log(task)
        self.assertEqual(first['text'], '中')
        with (path / 'output.log').open('ab') as stream:
            stream.write('文'.encode()[-1:])
        second = self.queue.log(task,first['offset'])
        self.assertEqual(second['text'],'文')
        (path / 'output.log').write_bytes(b'x'*300000)
        self.assertTrue(self.queue.log(task)['truncated'])
        self.queue.action(task,'cancel')

    def test_http_page_api_and_cross_origin_protection(self):
        server = ThreadingHTTPServer(('127.0.0.1',0), make_handler(self.queue,0))
        port = server.server_address[1]
        server.RequestHandlerClass = make_handler(self.queue,port)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{port}'
        try:
            page = urlopen(base).read().decode()
            import re
            token = re.search(r'name="queue-token" content="([^"]+)"',page).group(1)
            headers={'X-Queue-Token':token,'Content-Type':'application/json'}
            request = Request(base+'/api/tasks',data=json.dumps({'command':'echo http','gpu':'cpu'}).encode(),headers=headers)
            result = json.load(urlopen(request))
            self.assertEqual(self.task(result['id'])['command'],'echo http')
            for bad_headers in ({}, {**headers,'Origin':'https://evil.example'}, {**headers,'Host':'evil.example'}):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base+'/api/state',headers=bad_headers))
                self.assertEqual(error.exception.code,403)
            self.assertEqual(urlopen(base+'/app.js').status,200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_cli_submits_exact_commands_and_manages_shared_queue(self):
        server = ThreadingHTTPServer(('127.0.0.1',0), make_handler(self.queue,0))
        port = server.server_address[1]
        server.RequestHandlerClass = make_handler(self.queue,port)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        cli = Path(__file__).resolve().parents[1] / 'queue_cli.py'

        def call(*args, stdin=None, success=True):
            result = subprocess.run([sys.executable,str(cli),'--port',str(port),*args],
                                    input=stdin,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0 if success else 1,result.stderr)
            return json.loads(result.stdout if success else result.stderr)

        try:
            command = "printf '%s\\n' '中文; $HOME'\necho done"
            added = call('add','--gpu','cpu','--cwd',str(self.root),'--command',command)
            self.assertEqual(self.task(added['id'])['command'], command)
            from_stdin = call('add','--gpu','cpu','--file','-',stdin='echo stdin')
            self.assertEqual(self.task(from_stdin['id'])['command'],'echo stdin')
            file = self.root / 'commands.txt'
            file.write_text('echo 文件输入',encoding='utf-8')
            from_file = call('add','--gpu','cpu','--file',str(file))
            self.assertEqual(self.task(from_file['id'])['command'],'echo 文件输入')
            call('top',str(from_file['id']))
            self.assertEqual(call('status')['tasks'][0]['id'],from_file['id'])
            self.assertEqual(call('log',str(added['id']))['text'],'')
            call('pause')
            self.assertTrue(self.queue.settings()['paused'])
            call('resume')
            self.assertFalse(self.queue.settings()['paused'])
            call('cancel',str(added['id']))
            retry = call('retry',str(added['id']))
            self.assertNotEqual(retry['id'],added['id'])
            self.assertEqual(self.task(retry['id'])['command'],command)
            count = len(self.queue.state()['tasks'])
            error = call('add','--command','echo invalid','--gpu','not-a-gpu',success=False)
            self.assertIn('GPU',error['error'])
            self.assertEqual(len(self.queue.state()['tasks']),count)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_external_process_http_and_cli_end_to_end(self):
        script=self.root/'train_external.py'
        script.write_text("import time\nprint('Epoch [4/20] loss: 0.21',flush=True)\ntime.sleep(60)\n")
        log=self.root/'external.log'
        output=log.open('wb')
        child=subprocess.Popen([sys.executable,str(script)],stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.queue,0))
        port=server.server_address[1]
        server.RequestHandlerClass=make_handler(self.queue,port)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        self.queue.process_monitor.gpu_probe=lambda:([],None)
        from queue_cli import Client
        client=Client(port)
        cli=Path(__file__).resolve().parents[1]/'queue_cli.py'

        def call(*args):
            result=subprocess.run([sys.executable,str(cli),'--port',str(port),*args],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            return json.loads(result.stdout)

        try:
            self.wait_for(lambda:log.exists() and bool(log.read_text()))
            rows=call('processes')['processes']
            task=next(p for p in rows if p['pid']==child.pid)
            identity=task['identity']
            self.assertEqual(task['progress']['percent'],20)
            self.assertIn('Epoch',call('process-log',identity)['text'])
            call('bind-log',identity,str(log))
            self.assertEqual(next(p for p in client.request('/api/processes')['processes'] if p['identity']==identity)['log_source'],'manual')
            call('process-control',identity,'pause')
            self.assertEqual(next(p for p in client.request('/api/processes')['processes'] if p['pid']==child.pid)['status'],'paused')
            call('process-control',identity,'resume')
            call('process-control',identity,'stop')
            child.wait(timeout=4)
            self.assertEqual(len(self.queue.state()['tasks']),0)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGCONT)
                child.kill()
            child.wait()
            output.close()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
