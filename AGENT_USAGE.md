# Train Queue：供 Codex、Claude Code 使用的本机训练队列

## 位置与用途

- 项目：`/data/xjc/train-queue`
- 命令行入口：`python3 /data/xjc/train-queue/queue_cli.py`
- 详细说明：`/data/xjc/train-queue/README.md`
- 默认服务：`http://127.0.0.1:8765`

当用户要求将训练加入队列、依次运行多次实验、查看或管理已有训练、或使用 Train Queue 时，通过命令行入口操作。网页与命令行共用任务记录。一般代码修改无需访问队列。

## 操作步骤

1. 查看当前服务与任务：

   ```bash
   python3 /data/xjc/train-queue/queue_cli.py status
   ```

   若连接失败，说明默认端口的服务尚不可达。先检查已有服务及用户是否指定其他端口，避免启动重复服务。默认服务的启动命令是：

   ```bash
   python3 /data/xjc/train-queue/app.py --cwd /data/xjc
   ```

2. 从当前训练项目确认实际启动命令、项目目录和 Python / Conda 环境。工作目录必须是训练项目目录。优先用训练环境的 Python 绝对路径，或在同一个任务里初始化 Conda。根据用户要求指定物理 GPU；未指定时使用 `auto`。

3. 提交一个完整任务：

   ```bash
   python3 /data/xjc/train-queue/queue_cli.py add \
     --cwd /训练项目的绝对路径 \
     --gpu auto \
     --command '/训练环境/bin/python -u train.py --seed 1'
   ```

   上述路径与命令是示例，提交前替换为项目实际值。对于多行和复杂引号，先把完整命令写进 UTF-8 文件，再使用 `--file /绝对路径/command.txt`；也可用 `--file -` 从标准输入提交。每次提交创建一个任务；需要三个独立实验时分别调用三次。

4. 成功后记录返回的 `id` 并报告给用户。可再调用 `status` 确认。入队成功不代表训练已经启动或完成。提交超时或连接意外中断时先查询状态确认是否已入队，避免直接重复提交。

## 查询与管理

```bash
python3 /data/xjc/train-queue/queue_cli.py status
python3 /data/xjc/train-queue/queue_cli.py log 任务编号
python3 /data/xjc/train-queue/queue_cli.py cancel 任务编号
python3 /data/xjc/train-queue/queue_cli.py retry 任务编号
python3 /data/xjc/train-queue/queue_cli.py top 任务编号
python3 /data/xjc/train-queue/queue_cli.py pause
python3 /data/xjc/train-queue/queue_cli.py resume
```

只在用户任务范围内管理相关任务。`retry` 创建新任务，保留旧记录。`pause` 只暂停后续调度，运行中的训练继续。

## 已在运行的训练（此前从其他终端启动）

```bash
python3 /data/xjc/train-queue/queue_cli.py processes
python3 /data/xjc/train-queue/queue_cli.py processes --all
python3 /data/xjc/train-queue/queue_cli.py process-log 完整进程身份
python3 /data/xjc/train-queue/queue_cli.py bind-log 完整进程身份 /绝对路径/train.log
python3 /data/xjc/train-queue/queue_cli.py process-control 完整进程身份 track
python3 /data/xjc/train-queue/queue_cli.py process-control 完整进程身份 pause
python3 /data/xjc/train-queue/queue_cli.py process-control 完整进程身份 resume
python3 /data/xjc/train-queue/queue_cli.py process-control 完整进程身份 stop
python3 /data/xjc/train-queue/queue_cli.py process-control 完整进程身份 kill
```

完整进程身份取自 `processes` 输出的 `identity`，包含 PID、启动时间和本次系统启动标识，不要只凭 PID 猜测或拼接。`processes` 默认显示自动识别的训练，`--all` 可找出未识别进程，`track` 将其纳入监控。查询不会暂停、停止或重跑训练。只在用户明确要求操作的目标范围内使用进程控制；`stop` 请求终止，`kill` 才是强制结束。暂停不会释放 GPU 显存。训练进度来自日志，未知总步数时不推算百分比；外部进程结束后不能追溯退出码。

服务尚未启动时，可用 `python3 /data/xjc/train-queue/process_monitor.py` 只读检查当前训练。服务和训练需要处于同一进程可见空间，不能在隔离沙箱中查看宿主机任务。

## 运行约定

- 所有命令返回 JSON；失败时退出码非零，错误 JSON 写入标准错误。
- 自定义服务端口时，在子命令前加 `--port 端口`，例如 `queue_cli.py --port 8877 status`。
- 严格按顺序运行，一次一个任务；队首等 GPU 时后续任务也等待。
- `auto` 自动等一张空闲显卡；`0,1` 等待两张指定卡同时空闲；`cpu` 用于 CPU 命令或调试。
- 任务以环境变量分配 GPU，不要在训练命令中重新覆盖 `CUDA_VISIBLE_DEVICES`。分到一张物理 GPU 后，训练代码通常仍用 `cuda:0`。
- 使用前台训练命令，不要通过 `nohup`、`&`、`tmux` 或 `setsid` 把任务移出执行器的管理范围。
- 每个任务环境独立，前一任务中的 `cd`、Conda 激活和变量不会传给下一条。
- 本机 CLI 需要能够访问这台训练服务器；会话本身的文件、网络和命令权限仍适用。
