import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from process_monitor import ProcessMonitor, parse_progress, training_command, pidfd_open


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.monitor = ProcessMonitor(self.root,gpu_probe=lambda:([],None))
        self.children = []
        self.handles = []

    def tearDown(self):
        for process in self.children:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGCONT)
                os.killpg(process.pid,signal.SIGKILL)
            else:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            if process.stdout:
                process.stdout.close()
        for handle in self.handles:
            handle.close()
        self.tmp.cleanup()

    def wait(self, condition,timeout=5):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if condition():
                return
            time.sleep(.05)
        self.fail('Timed out')

    def spawn(self,name='train_existing.py',tree=False,pipe=False,ignore_term=False):
        script=self.root/name
        script.write_text("import os, subprocess, time, signal\n"+
                          ("signal.signal(signal.SIGTERM,signal.SIG_IGN)\n" if ignore_term else '')+
                          ("child=subprocess.Popen(['sleep','60']); print('child='+str(child.pid),flush=True)\n" if tree else '')+
                          "print('Epoch [3/10] step: 30/100 loss: 0.25 lr: 1e-4',flush=True)\n"+
                          "time.sleep(60)\n")
        log=self.root/(name+'.log')
        output=subprocess.PIPE if pipe else log.open('wb')
        if not pipe:
            self.handles.append(output)
        process=subprocess.Popen([sys.executable,str(script)],stdout=output,stderr=subprocess.STDOUT,
                                 cwd=self.root,start_new_session=True)
        self.children.append(process)
        if not pipe:
            self.wait(lambda:log.exists() and 'loss' in log.read_text())
        else:
            time.sleep(.15)
        identity=self.monitor.read(process.pid)['identity']
        return process,identity,log

    def test_existing_training_detected_with_log_and_progress(self):
        process,identity,log=self.spawn(tree=True)
        row=next(p for p in self.monitor.snapshot(force=True)['processes'] if p['identity']==identity)
        self.assertEqual(row['pid'],process.pid)
        self.assertEqual(len(row['members']),2)
        self.assertEqual(row['progress']['current'],30)
        self.assertEqual(row['progress']['total'],100)
        self.assertEqual(row['progress']['percent'],30)
        self.assertEqual(row['progress']['metrics']['loss'],'0.25')
        self.assertEqual(row['log_path'],str(log))
        self.assertGreaterEqual(row['elapsed_seconds'],0)
        self.assertIn('Epoch',self.monitor.logs(identity)['text'])

    def test_pipe_output_not_consumed_and_manual_binding_persists(self):
        process,identity,_=self.spawn(pipe=True)
        result=self.monitor.logs(identity)
        self.assertEqual(result['text'],'')
        self.assertIn('没有可读取',result['note'])
        # Monitoring must not steal a byte from the program's existing output pipe.
        self.assertIn(b'Epoch',process.stdout.readline())
        log=self.root/'manual.log'
        log.write_text('iteration: 25/200 loss=0.17\n')
        self.monitor.bind(identity,str(log))
        restarted=ProcessMonitor(self.root,gpu_probe=lambda:([],None))
        self.assertIn('loss=0.17',restarted.logs(identity)['text'])
        row=next(p for p in restarted.snapshot(force=True)['processes'] if p['identity']==identity)
        self.assertEqual(row['progress']['percent'],12.5)

    def test_unrecognized_process_can_be_tracked(self):
        process,identity,_=self.spawn(name='experiment.py')
        self.assertNotIn(identity,[p['identity'] for p in self.monitor.snapshot(force=True)['processes']])
        self.assertIn(identity,[p['identity'] for p in self.monitor.snapshot(mode='all')['processes']])
        self.monitor.bind(identity)
        self.assertIn(identity,[p['identity'] for p in self.monitor.snapshot(force=True)['processes']])

    def test_pause_resume_stop_tree_and_external_history(self):
        process,identity,log=self.spawn(tree=True)
        self.monitor.snapshot(force=True)
        members=[p['pid'] for p in next(p for p in self.monitor.cached['processes'] if p['identity']==identity)['members']]
        result=self.monitor.control(identity,'pause')
        self.assertEqual(set(result['sent_pids']),set(members))
        self.wait(lambda:all(self.monitor.read(pid)['state'] in ('T','t') for pid in members))
        paused=next(p for p in self.monitor.snapshot(force=True)['processes'] if p['identity']==identity)
        self.assertEqual(paused['status'],'paused')
        self.monitor.control(identity,'resume')
        self.wait(lambda:all(self.monitor.read(pid)['state'] not in ('T','t') for pid in members))
        self.monitor.control(identity,'stop')
        process.wait(timeout=4)
        self.wait(lambda:not any(self.monitor.read(pid) for pid in members))
        history=self.monitor.snapshot(force=True)['history']
        self.assertEqual(next(p for p in history if p['identity']==identity)['status'],'exited')
        self.assertIn('loss',self.monitor.logs(identity)['text'])

    def test_stop_does_not_automatically_kill_term_ignoring_process(self):
        process,identity,_=self.spawn(ignore_term=True)
        self.monitor.control(identity,'stop')
        time.sleep(.15)
        self.assertIsNone(process.poll())
        self.monitor.control(identity,'kill')
        self.assertEqual(process.wait(timeout=4),-signal.SIGKILL)

    def test_stale_pid_and_service_ancestors_and_managed_tasks_rejected(self):
        process,identity,_=self.spawn()
        for target,excluded in [(identity+'old',()),(self.monitor.read(os.getpid())['identity'],()),(identity,(process.pid,))]:
            with self.assertRaises(ValueError),patch('process_monitor.can_control',return_value=True),patch('process_monitor.pidfd_send') as send:
                self.monitor.control(target,'stop',excluded)
            send.assert_not_called()
        self.assertIsNone(process.poll())

    def test_pid_reused_between_validation_and_signal_is_rejected(self):
        process,identity,_=self.spawn()
        original_open=pidfd_open
        original_read=self.monitor.read
        changed=False

        def open_then_change(pid):
            nonlocal changed
            descriptor=original_open(pid)
            changed=True
            return descriptor

        def read_changed(pid):
            value=original_read(pid)
            if changed and pid==process.pid:
                value={**value,'identity':identity+'different'}
            return value

        with patch('process_monitor.can_control',return_value=True),patch('process_monitor.pidfd_open',side_effect=open_then_change),patch.object(self.monitor,'read',side_effect=read_changed),patch('process_monitor.pidfd_send') as send:
            result=self.monitor.control(identity,'stop')
        send.assert_not_called()
        self.assertIn(process.pid,result['vanished_pids'])

    def test_other_user_not_discovered_or_controllable(self):
        fake_proc=self.root/'proc'
        (fake_proc/'99').mkdir(parents=True)
        (fake_proc/'99/status').write_text(f'Name:\ttrain\nUid:\t{os.getuid()+1}\t{os.getuid()+1}\t0\t0\n')
        self.monitor.proc=fake_proc
        self.assertIsNone(self.monitor.read(99))
        with self.assertRaises(ValueError),patch('process_monitor.can_control',return_value=True),patch('process_monitor.pidfd_send') as send:
            self.monitor.control('99-1-'+self.monitor.boot_id,'stop')
        send.assert_not_called()

    def test_gpu_process_detected_even_without_training_name(self):
        process,identity,_=self.spawn(name='experiment.py')
        self.monitor.gpu_probe=lambda:([{'pid':process.pid,'uuid':'GPU-test','memory_mb':2000}],None)
        row=next(p for p in self.monitor.snapshot(force=True,gpu_names={'GPU-test':'7'})['processes'] if p['identity']==identity)
        self.assertEqual(row['gpus'][0]['index'],'7')
        self.assertEqual(row['gpus'][0]['memory_mb'],2000)

    def test_regular_files_only(self):
        _,identity,_=self.spawn(pipe=True)
        fifo=self.root/'pipe'
        os.mkfifo(fifo)
        for path in (fifo,Path('/dev/null'),self.root):
            with self.assertRaises(ValueError):
                self.monitor.bind(identity,str(path))

    def test_training_command_and_progress_parsing(self):
        for argv in (['python','-u','train.py'],['python','-m','torch.distributed.run'],['torchrun','x.py'],['python','finetune_model.py']):
            self.assertTrue(training_command(argv),argv)
        for argv in (['python','queue_cli.py','--command','/tmp/train.py'],['python','-c','train.py'],['bash','-c','python train.py']):
            self.assertFalse(training_command(argv),argv)
        self.assertEqual(parse_progress('Epoch [1/10]\rEpoch [2/10]')['percent'],20)
        self.assertEqual(parse_progress(' 15%|###| 150/1000 [00:03<00:30, 5.1it/s]')['percent'],15)
        self.assertIsNone(parse_progress('iter: 12,345 loss: 0.123')['percent'])
        self.assertEqual(parse_progress('iter: 12,345 loss: 0.123')['current'],12345)
        json_progress=parse_progress('{"step":711,"loss":0.017,"lr":0.00005}\n', ['python','train.py','--steps','400000'])
        self.assertEqual(json_progress['current'],711)
        self.assertEqual(json_progress['total'],400000)
        self.assertEqual(json_progress['percent'],0.18)
        self.assertEqual(json_progress['metrics']['loss'],'0.017')
        self.assertIsNone(parse_progress('{"step":9000,"total":0.0017}')['percent'])


if __name__=='__main__':
    unittest.main()
