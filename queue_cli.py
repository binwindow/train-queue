"""Local command-line client for humans and coding agents. Requires a running server."""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import build_opener, ProxyHandler, Request


class SessionParser(HTMLParser):
    token = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta' and attrs.get('name') == 'queue-token':
            self.token = attrs.get('content')


class Client:
    def __init__(self, port=8765):
        self.base = f'http://127.0.0.1:{port}'
        # Local requests must not send the session token through an HTTP proxy.
        self.opener = build_opener(ProxyHandler({}))
        self.token = None

    def request(self, path, data=None):
        if self.token is None:
            with self.opener.open(self.base + '/', timeout=10) as response:
                parser = SessionParser()
                parser.feed(response.read().decode('utf-8'))
            if not parser.token:
                raise ValueError('未能读取队列服务会话，请确认端口对应 Train Queue')
            self.token = parser.token
        headers = {'X-Queue-Token': self.token}
        body = None
        if data is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        request = Request(self.base + path, data=body, headers=headers)
        with self.opener.open(request, timeout=30) as response:
            return json.load(response)


def main(argv=None):
    parser = argparse.ArgumentParser(description='向本机 Train Queue 提交训练任务或查看状态；结果为 JSON')
    parser.add_argument('--port', type=int, default=8765, help='本机队列服务端口，默认 8765')
    commands = parser.add_subparsers(dest='action', required=True)
    add = commands.add_parser('add', help='将一个完整命令加入队列')
    source = add.add_mutually_exclusive_group(required=True)
    source.add_argument('-c', '--command', help='训练启动命令')
    source.add_argument('-f', '--file', help='从 UTF-8 文件读取命令；- 表示标准输入')
    add.add_argument('--cwd', help='服务器上的项目目录；省略时使用服务默认目录')
    add.add_argument('--gpu', default='auto', help='auto、cpu 或物理 GPU 编号，如 0,1')
    add.add_argument('--min-free-mb', type=int, default=1024, help='所需空闲显存，MiB')
    commands.add_parser('status', help='查看全部任务、设置和 GPU 状态')
    log = commands.add_parser('log', help='读取任务日志（最多最近 256 KiB）')
    log.add_argument('id', type=int)
    log.add_argument('--offset', type=int, default=0)
    for action, help_text in [('cancel', '取消任务'), ('retry', '创建一条相同的新任务'), ('top', '将等待任务置顶')]:
        command = commands.add_parser(action, help=help_text)
        command.add_argument('id', type=int)
    commands.add_parser('pause', help='暂停后续调度，当前训练继续')
    commands.add_parser('resume', help='恢复队列调度')
    processes = commands.add_parser('processes',help='查看当前用户已在运行的训练进程')
    processes.add_argument('--all',action='store_true',help='包含未自动识别的用户进程')
    process_log = commands.add_parser('process-log',help='读取已有训练进程日志')
    process_log.add_argument('identity',help='processes 返回的完整进程身份')
    process_log.add_argument('--offset',type=int,default=0)
    process_control = commands.add_parser('process-control',help='管理当前用户已有训练进程及其子进程')
    process_control.add_argument('identity')
    process_control.add_argument('operation',choices=['stop','kill','pause','resume','track'])
    bind_log = commands.add_parser('bind-log',help='为已有进程关联一个当前用户拥有的日志文件')
    bind_log.add_argument('identity')
    bind_log.add_argument('path')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('端口必须在 1～65535 之间')
    client = Client(args.port)
    try:
        if args.action == 'add':
            command = args.command
            if args.file is not None:
                command = sys.stdin.read() if args.file == '-' else Path(args.file).read_text(encoding='utf-8')
            payload = dict(command=command, gpu=args.gpu, min_free_mb=args.min_free_mb)
            if args.cwd is not None:
                payload['cwd'] = args.cwd
            result = client.request('/api/tasks', payload)
        elif args.action == 'status':
            result = client.request('/api/state')
        elif args.action == 'log':
            result = client.request(f'/api/tasks/{args.id}/log?offset={args.offset}')
        elif args.action in ('pause', 'resume'):
            result = client.request('/api/settings', {'paused': args.action == 'pause'})
        elif args.action == 'processes':
            result = client.request('/api/processes?mode='+('all' if args.all else 'training'))
        elif args.action == 'process-log':
            result = client.request(f'/api/processes/{args.identity}/log?offset={args.offset}')
        elif args.action == 'bind-log':
            result = client.request(f'/api/processes/{args.identity}/bind-log',{'path':args.path})
        elif args.action == 'process-control':
            result = client.request(f'/api/processes/{args.identity}/{args.operation}',{})
        else:
            result = client.request(f'/api/tasks/{args.id}/{args.action}', {})
    except HTTPError as exc:
        try:
            message = json.load(exc).get('error', str(exc))
        except (ValueError, AttributeError):
            message = str(exc)
        print(json.dumps({'error': message}, ensure_ascii=False), file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        message = str(exc)
        if isinstance(exc, URLError):
            message = f'无法连接本机队列服务 {client.base}，请先启动 app.py。详情：{exc.reason}'
        print(json.dumps({'error': message}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
