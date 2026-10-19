"""Observe existing user processes without attaching a debugger or consuming pipes."""
import csv
import ctypes
import errno
import io
import json
import os
from pathlib import Path
import pwd
import platform
import re
import shlex
import signal
import stat
import subprocess
import threading
import time

from worker import atomic_json


def kernel_call(number, *args):
    # Some Conda Python builds lack pidfd wrappers even on a capable Linux kernel.
    libc = ctypes.CDLL(None,use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number),*args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error,os.strerror(error))
    return result


def pidfd_open(pid):
    if hasattr(os,'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.machine() not in ('x86_64','aarch64'):
        raise OSError(errno.ENOSYS,'Unsupported pidfd architecture')
    return kernel_call(434,ctypes.c_int(pid),ctypes.c_uint(0))


def pidfd_send(descriptor, sig):
    if hasattr(signal,'pidfd_send_signal'):
        return signal.pidfd_send_signal(descriptor,sig)
    if platform.machine() not in ('x86_64','aarch64'):
        raise OSError(errno.ENOSYS,'Unsupported pidfd architecture')
    return kernel_call(424,ctypes.c_int(descriptor),ctypes.c_int(sig),ctypes.c_void_p(0),ctypes.c_uint(0))


def can_control():
    try:
        descriptor = pidfd_open(os.getpid())
        try:
            pidfd_send(descriptor,0)
        finally:
            os.close(descriptor)
        return True
    except OSError:
        return False


def parse_progress(text, argv=None):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    result = {'current': None, 'total': None, 'percent': None, 'unit': None, 'metrics': {}}
    number = r'[\d,]+'
    patterns = [
        (rf'\b(epoch|step|iter(?:ation)?s?)\s*[:=\[]?\s*({number})\s*/\s*({number})', None),
        (rf'\b(epoch|step|iter(?:ation)?s?)\s*[:=\[]\s*({number})\s*\]\s*\[\s*({number})\s*\]', None),
        (rf'({number})\s*/\s*({number})\s*\[[^\]\r\n]*(?:it/s|s/it|<)', 'iteration'),
    ]
    # Use the most recent explicit progress entry; do not infer totals from utilization.
    latest = None
    for pattern, default_unit in patterns:
        for match in re.finditer(pattern, text, re.I):
            unit, current, total = ((match.group(1).lower(),match.group(2),match.group(3))
                                    if default_unit is None else (default_unit,match.group(1),match.group(2)))
            current, total = int(current.replace(',', '')), int(total.replace(',', ''))
            if total > 0 and 0 <= current <= total and (latest is None or match.start() >= latest[0]):
                latest = match.start(), unit, current, total
    structured_latest = None
    for line in re.finditer(r'[^\r\n]+',text):
        value = line.group()
        start = value.find('{')
        if start < 0:
            continue
        try:
            record = json.loads(value[start:])
        except ValueError:
            continue
        if not isinstance(record,dict):
            continue
        for unit, names, total_names in (
            ('step',('step','global_step','iteration','iter'),('total_steps','max_steps','num_train_steps','num_training_steps','total_iters','max_iters')),
            ('epoch',('epoch',),('total_epochs','num_epochs','epochs')),
        ):
            current = next((record[key] for key in names if key in record),None)
            total = next((record[key] for key in total_names if key in record),None)
            if isinstance(current,int) and not isinstance(current,bool) and current >= 0:
                if not (isinstance(total,int) and not isinstance(total,bool) and total>0 and total>=current):
                    total=None
                structured_latest=(line.start(),unit,current,total)
                break
        metrics=record.get('metrics',{})
        metrics={**record,**(metrics if isinstance(metrics,dict) else {})}
        for key,value in metrics.items():
            if re.fullmatch(r'loss[\w/.-]*|lr|psnr|ssim|acc(?:uracy)?|total|grad_norm|seconds_per_step',key,re.I) and isinstance(value,(int,float)) and not isinstance(value,bool):
                result['metrics'][key]=str(value)
    if structured_latest and (latest is None or structured_latest[0]>=latest[0]):
        latest=structured_latest
    if latest:
        _, unit, current, total = latest
        result.update(unit=unit,current=current,total=total,percent=round(current/total*100,2) if total else None)
    else:
        entries = list(re.finditer(rf'\b(epoch|step|iter(?:ation)?s?)\s*[:=\[]\s*({number})',text,re.I))
        if entries:
            match = entries[-1]
            result.update(unit=match.group(1).lower(),current=int(match.group(2).replace(',','')))
    for match in re.finditer(r'\b(loss[\w/.-]*|lr|psnr|ssim|acc(?:uracy)?)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',text,re.I):
        result['metrics'][match.group(1).lower()] = match.group(2)
    if result['current'] is not None and result['total'] is None and argv:
        step_flags={'--steps','--total-steps','--max-steps','--train-steps','--training-steps','--num-training-steps','--num-train-steps','--max-train-steps','--total-iters','--max-iters','--iterations'}
        epoch_flags={'--epochs','--num-epochs','--total-epochs','--max-epochs'}
        allowed=epoch_flags if result['unit']=='epoch' else step_flags
        for i,arg in enumerate(argv):
            flag,separator,value=arg.partition('=')
            if flag.replace('_','-') not in allowed:
                continue
            if not separator:
                value=argv[i+1] if i+1<len(argv) else ''
            if value.replace(',','').replace('_','').isdigit():
                total=int(value.replace(',','').replace('_',''))
                if total>0 and total>=result['current']:
                    result.update(total=total,percent=round(result['current']/total*100,2),total_source='command')
    lines = [line.strip() for line in re.split(r'[\r\n]',text) if line.strip()]
    result['last_line'] = lines[-1][-500:] if lines else ''
    return result


def gpu_processes():
    try:
        result = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                 '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
        rows = []
        mapping = subprocess.run(['nvidia-smi','--query-gpu=uuid,index','--format=csv,noheader,nounits'],
                                 capture_output=True,text=True,timeout=3,check=True)
        indices={row[0].strip():row[1].strip() for row in csv.reader(io.StringIO(mapping.stdout)) if len(row)==2}
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) == 3 and row[0].strip().isdigit():
                rows.append({'pid':int(row[0]),'uuid':row[1].strip(),'index':indices.get(row[1].strip(),row[1].strip()),
                             'memory_mb':int(row[2]) if row[2].strip().isdigit() else None})
        return rows, None
    except (OSError, ValueError, subprocess.SubprocessError):
        return [], '无法读取 GPU 进程，仍可通过训练命令识别；GPU 显存信息暂不可用'


def training_command(argv):
    if not argv:
        return False
    executable = Path(argv[0]).name.lower()
    if executable in ('torchrun','accelerate','deepspeed','horovodrun'):
        return True
    if not (executable.startswith('python') or executable.startswith('pypy')):
        return False
    # Examine the entrypoint/module, never a -c command body or an option value.
    arguments = iter(argv[1:])
    for arg in arguments:
        if arg == '-c' or arg.startswith('-c'):
            return False
        if arg in ('-W','-X'):
            next(arguments,None)
            continue
        if arg == '-m':
            arg = next(arguments,'')
        elif arg.startswith('-'):
            continue
        return bool(re.search(r'(?:^|[_./-])(train(?:ing)?|finetun\w*|pretrain\w*)(?:[_./-]|$)',
                              Path(arg).name.lower()) or arg in ('torch.distributed.run','torch.distributed.launch',
                              'accelerate.commands.launch','deepspeed.launcher.runner'))
    return False


class ProcessMonitor:
    def __init__(self, data_dir, proc_root='/proc', gpu_probe=gpu_processes):
        self.proc = Path(proc_root)
        self.uid = os.getuid()
        self.username = pwd.getpwuid(self.uid).pw_name
        self.ticks = os.sysconf('SC_CLK_TCK')
        self.pagesize = os.sysconf('SC_PAGE_SIZE')
        self.boot_id = (self.proc/'sys/kernel/random/boot_id').read_text().strip().replace('-','')
        self.storage = Path(data_dir)/'process-bindings.json'
        try:
            self.bindings = json.loads(self.storage.read_text())
        except (OSError, ValueError):
            self.bindings = {}
        self.gpu_probe = gpu_probe
        self.lock = threading.RLock()
        self.last_scan = 0
        self.previous = {}
        self.known = {}
        self.requests = {}
        self.cached = {}
        self.cached_excluded = None

    def read(self, pid):
        try:
            directory = self.proc/str(pid)
            status = (directory/'status').read_text()
            uids = re.search(r'^Uid:\s+(\d+)\s+(\d+)',status,re.M)
            if not uids or any(int(i) != self.uid for i in uids.groups()):
                return None
            fields = (directory/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0] in ('Z','X'):
                return None
            start = fields[19]
            argv = [part.decode('utf-8',errors='replace') for part in (directory/'cmdline').read_bytes().split(b'\0') if part]
            if not argv:
                return None
            try:
                cwd = os.readlink(directory/'cwd')
            except OSError:
                cwd = ''
            return dict(pid=int(pid),ppid=int(fields[1]),pgid=int(fields[2]),start=start,
                        identity=f'{pid}-{start}-{self.boot_id}',state=fields[0],argv=argv,
                        command=shlex.join(argv),cwd=cwd,cpu_ticks=int(fields[11])+int(fields[12]),
                        rss_mb=max(0,int(fields[21]))*self.pagesize/1024**2)
        except (OSError, ValueError, IndexError):
            return None

    def all_processes(self):
        return {int(item.name): process for item in self.proc.iterdir() if item.name.isdigit()
                if (process := self.read(int(item.name))) is not None}

    @staticmethod
    def descendants(pid, processes):
        found, pending = {pid}, [pid]
        children = {}
        for child, process in processes.items():
            children.setdefault(process['ppid'],[]).append(child)
        while pending:
            for child in children.get(pending.pop(),[]):
                if child not in found:
                    found.add(child)
                    pending.append(child)
        return found

    def protected(self, processes, excluded):
        protected = {1, os.getpid()}
        pid = os.getpid()
        while pid in processes:
            parent = processes[pid]['ppid']
            if parent in protected:
                break
            protected.add(parent)
            pid = parent
        for pid in excluded:
            protected.update(self.descendants(pid,processes))
        return protected

    def validate(self, identity, excluded=()):
        try:
            pid = int(identity.split('-')[0])
        except (AttributeError,ValueError):
            raise ValueError('无效的进程身份')
        process = self.read(pid)
        if not process or process['identity'] != identity:
            raise ValueError('进程已退出、身份已变化，或不属于当前用户；请刷新')
        processes = self.all_processes()
        if pid in self.protected(processes,excluded):
            raise ValueError('该进程属于工作台、其父进程或队列执行器，请使用队列任务操作')
        return process, processes

    def regular_file(self, path):
        info = os.stat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != self.uid:
            raise ValueError('仅可读取当前用户拥有的普通日志文件')
        return info

    def find_log(self, process, members):
        binding = self.bindings.get(process['identity'],{})
        if binding.get('log_path'):
            return binding['log_path'], 'manual'
        for member in [process] + [p for p in members if p['pid'] != process['pid']]:
            for fd in ('1','2'):
                path = self.proc/str(member['pid'])/'fd'/fd
                try:
                    self.regular_file(path)
                    target = os.readlink(path)
                    if not target.endswith(' (deleted)'):
                        return str(path), 'stdout' if fd == '1' else 'stderr'
                except (OSError,ValueError):
                    pass
        return None, None

    def read_log(self, path, offset=0, limit=262144):
        if offset < 0:
            raise ValueError('日志偏移不能为负数')
        if not path:
            return dict(text='',offset=0,truncated=False,file_id=None,path=None,
                        note='没有可读取的日志。输出在终端或管道时，请关联已有日志文件。')
        # Nonblocking open plus fstat prevents a replaced path from hanging on a FIFO.
        fd = os.open(path,os.O_RDONLY|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != self.uid:
                raise ValueError('仅可读取当前用户拥有的普通日志文件')
            start = max(min(offset,info.st_size),info.st_size-limit)
            stream.seek(start)
            raw = stream.read(limit)
            try:
                text = raw.decode('utf-8')
            except UnicodeDecodeError as exc:
                if exc.reason == 'unexpected end of data' and exc.end == len(raw):
                    raw = raw[:exc.start]
                text = raw.decode('utf-8',errors='replace')
            display = os.readlink(path) if str(path).startswith(str(self.proc)+'/') else str(path)
            return dict(text=text,offset=start+len(raw),truncated=start>offset,
                        file_id=f'{info.st_dev}:{info.st_ino}',path=display,modified_at=info.st_mtime)

    def snapshot(self, excluded=(), mode='training', force=False, gpu_names=None):
        if mode not in ('training','all'):
            raise ValueError('进程筛选必须为 training 或 all')
        with self.lock:
            key = tuple(sorted(excluded))
            if force or key != self.cached_excluded or time.monotonic()-self.last_scan >= 2:
                now = time.monotonic()
                processes = self.all_processes()
                protected = self.protected(processes,excluded)
                processes = {pid:p for pid,p in processes.items() if pid not in protected}
                gpu_rows, gpu_error = self.gpu_probe()
                by_pid = {}
                for gpu in gpu_rows:
                    by_pid.setdefault(gpu['pid'],[]).append(gpu)
                candidates = {pid for pid,p in processes.items() if pid in by_pid or training_command(p['argv'])
                              or p['identity'] in self.bindings}
                roots = set(candidates)
                for pid in candidates:
                    ancestor, seen = processes[pid]['ppid'], {pid}
                    while ancestor in processes and ancestor not in seen:
                        if ancestor in candidates:
                            roots.discard(pid)
                            break
                        seen.add(ancestor)
                        ancestor = processes[ancestor]['ppid']
                grouped = set()
                entries = []
                uptime = float((self.proc/'uptime').read_text().split()[0])
                elapsed = now-self.last_scan if self.last_scan else 0
                for pid,p in processes.items():
                    old = self.previous.get(p['identity'])
                    p['cpu_percent'] = round(max(0,p['cpu_ticks']-old)/self.ticks/elapsed*100,1) if old is not None and elapsed>0 else None
                for pid in sorted(roots):
                    ids = self.descendants(pid,processes)
                    grouped.update(ids)
                    members = [processes[i] for i in ids if i in processes]
                    entries.append(self.describe(processes[pid],members,by_pid,uptime,True,gpu_names))
                for pid,p in sorted(processes.items()):
                    if pid not in grouped:
                        entries.append(self.describe(p,[p],by_pid,uptime,False,gpu_names))
                active_ids = {p['identity'] for p in entries}
                for entry in entries:
                    if entry['is_training']:
                        self.known[entry['identity']] = entry.copy()
                for identity, entry in self.known.items():
                    if identity not in active_ids and entry['status'] != 'exited':
                        # Child may now belong to a newly recognized parent group.
                        if entry['pid'] in processes:
                            continue
                        entry.update(status='exited',finished_at=time.time(),note='进程已结束；外部启动任务的退出码无法追溯')
                history = [p for p in self.known.values() if p['status']=='exited'][-100:]
                self.previous = {p['identity']:p['cpu_ticks'] for p in processes.values()}
                self.last_scan = now
                self.cached_excluded = key
                self.cached = dict(processes=entries,history=history,gpu_error=gpu_error,
                                   username=self.username,uid=self.uid,scanned_at=time.time(),
                                   control_available=can_control())
            result = self.cached.copy()
            result['processes'] = [p for p in result['processes'] if mode=='all' or p['is_training']]
            return result

    def describe(self, process, members, by_pid, uptime, is_training, gpu_names):
        gpus = []
        for member in members:
            for gpu in by_pid.get(member['pid'],[]):
                gpus.append({**gpu,'index':(gpu_names or {}).get(gpu['uuid'],gpu.get('index',gpu['uuid']))})
        path, source = self.find_log(process,members) if is_training else (None,None)
        try:
            log = self.read_log(path,limit=65536) if path else {}
            progress = parse_progress(log.get('text',''),process['argv'])
            log_error = None
        except (OSError,ValueError) as exc:
            log, progress, log_error = {}, parse_progress(''), str(exc)
        return dict(identity=process['identity'],pid=process['pid'],ppid=process['ppid'],
                    command=process['command'],cwd=process['cwd'],is_training=is_training,
                    status='paused' if all(p['state'] in ('T','t') for p in members) else self.requests.get(process['identity'],'running'),
                    elapsed_seconds=max(0,uptime-int(process['start'])/self.ticks),
                    started_at=time.time()-max(0,uptime-int(process['start'])/self.ticks),
                    members=[{'pid':p['pid'],'identity':p['identity']} for p in members],
                    rss_mb=round(sum(p['rss_mb'] for p in members),1),
                    cpu_percent=round(sum(p['cpu_percent'] or 0 for p in members),1) if any(p['cpu_percent'] is not None for p in members) else None,
                    gpus=gpus,log_path=log.get('path'),log_source=source,log_error=log_error,
                    log_modified_at=log.get('modified_at'),progress=progress)

    def logs(self, identity, offset=0, excluded=()):
        with self.lock:
            try:
                process, processes = self.validate(identity,excluded)
                members = [processes[i] for i in self.descendants(process['pid'],processes) if i in processes]
                path, _ = self.find_log(process,members)
            except ValueError:
                previous = self.known.get(identity)
                if not previous or previous.get('status') != 'exited':
                    raise
                path = self.bindings.get(identity,{}).get('log_path') or previous.get('log_path')
            try:
                return self.read_log(path,offset)
            except OSError as exc:
                raise ValueError(f'日志不可读取：{exc}')

    def bind(self, identity, path=None, excluded=()):
        with self.lock:
            self.validate(identity,excluded)
            value = {'tracked':True}
            if path:
                path = str(Path(path).expanduser().resolve())
                try:
                    self.regular_file(path)
                except OSError as exc:
                    raise ValueError(f'日志文件不可读取：{exc}')
                value['log_path'] = path
            self.bindings[identity] = value
            atomic_json(self.storage,self.bindings)
            self.last_scan = 0
            return {'ok':True}

    def control(self, identity, action, excluded=()):
        signals = {'stop':signal.SIGTERM,'kill':signal.SIGKILL,'pause':signal.SIGSTOP,'resume':signal.SIGCONT}
        if action not in signals:
            raise ValueError('不支持的进程操作')
        if not can_control():
            raise ValueError('当前内核或会话权限不支持 pidfd 安全控制，仍可查看任务')
        with self.lock:
            root, processes = self.validate(identity,excluded)
            protected = self.protected(processes,excluded)
            members = [processes[pid] for pid in self.descendants(root['pid'],processes)
                       if pid in processes and pid not in protected]
            # Freeze the parent first when suspending; signal recorded descendants
            # individually, never an unrelated shell's shared process group.
            members.sort(key=lambda p:p['pid'] != root['pid'])
            sent, vanished = [], []
            for process in members:
                descriptor = None
                try:
                    descriptor = pidfd_open(process['pid'])
                    current = self.read(process['pid'])
                    if not current or current['identity'] != process['identity']:
                        vanished.append(process['pid'])
                        continue
                    if action == 'stop' and current['state'] in ('T','t'):
                        pidfd_send(descriptor,signal.SIGCONT)
                    pidfd_send(descriptor,signals[action])
                    sent.append(process['pid'])
                except ProcessLookupError:
                    vanished.append(process['pid'])
                except OSError as exc:
                    if exc.errno in (errno.ENOSYS,errno.EINVAL):
                        raise ValueError('当前内核不支持安全进程控制')
                    raise ValueError(f'部分进程操作未完成；已发送 {sent}；错误：{exc}')
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
            if action in ('stop','kill'):
                self.requests[identity] = 'stopping'
            elif action == 'resume':
                self.requests.pop(identity,None)
            self.last_scan = 0
            return {'ok':True,'sent_pids':sent,'vanished_pids':vanished,'action':action,
                    'note':'暂停不会释放 GPU 显存' if action=='pause' else '信号已发送，后续状态以扫描结果为准'}


if __name__ == '__main__':
    # Read-only diagnostic, useful before starting the web server.
    import argparse
    parser = argparse.ArgumentParser(description='只读扫描当前用户的已有训练，不操作训练进程')
    parser.add_argument('--all',action='store_true',help='包括未识别为训练的用户进程')
    args = parser.parse_args()
    monitor = ProcessMonitor(Path(__file__).resolve().parent/'.data')
    snapshot = monitor.snapshot(mode='all' if args.all else 'training',force=True)
    print(json.dumps({'username':snapshot['username'],'gpu_error':snapshot['gpu_error'],
                      'control_available':snapshot['control_available'],
                      'processes':[{key:p[key] for key in ('pid','identity','cwd','status','elapsed_seconds',
                                   'log_path','members','gpus','progress')} for p in snapshot['processes']]},
                     ensure_ascii=False,indent=2))
