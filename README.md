# Train Queue · 训练任务工作台

在类似对话框的网页里粘贴训练启动命令，按 **Enter** 提交。后台等待 GPU 空闲，按队列顺序执行，一次一个任务。支持实时日志、取消、置顶、重新排队、失败后暂停以及重启恢复。

这是 Linux / NVIDIA 服务器上的本机工具。后端和界面只使用 Python 标准库与原生 HTML/CSS/JavaScript，**不需要安装依赖、不需要大模型 API**。

## 启动

需要 Linux、Python 3.10+、Bash；GPU 调度需要服务器的 NVIDIA 驱动和 `nvidia-smi` 正常工作。

```bash
python3 /data/xjc/train-queue/app.py --cwd /data/xjc
```

在浏览器打开 **http://127.0.0.1:8765**。

如果浏览器在你的电脑上、程序运行在远程训练服务器上，在**电脑终端**建立转发：

```bash
ssh -L 8765:127.0.0.1:8765 用户名@训练服务器
```

保持 SSH 连接，在电脑浏览器访问同一个地址。SSH 使用不同端口时添加 `-p 端口号`。网页只监听服务器回环地址，不需要对外开放 8765 端口。SSH 转发后的本地端口请与服务端一致。

更改端口或数据目录：

```bash
python3 /data/xjc/train-queue/app.py --port 8877 --cwd /data/xjc --data-dir /data/xjc/train-queue/.data
```

若要在断开 SSH 后继续调度，建议放在 `tmux` 会话中运行：

```bash
tmux new -s train-queue
python3 /data/xjc/train-queue/app.py --cwd /data/xjc
```

按 `Ctrl+B`，再按 `D` 脱离会话。之后用 `tmux attach -t train-queue` 回到服务终端。

## 提交训练

1. 在“工作目录”中填写项目所在的**服务器路径**，例如 `/data/xjc/Flow_Matching_Same`。
2. 选择 GPU：`auto` 自动挑一张空闲显卡，`0` 指定物理卡 0，`0,1` 同时等待两张卡，`cpu` 用于普通命令和调试。
3. 填写启动命令，按 Enter 发送；Shift+Enter 换行。粘贴多行命令会作为一个完整任务执行。
4. 从任务列表打开日志，或调整等待队列。

推荐直接使用训练环境的 Python 绝对路径，避免服务所用 Python 与训练环境混淆：

```bash
/data/xjc/miniconda3/envs/你的环境/bin/python -u train.py --config configs/experiment.yaml
```

也可以在同一条多行任务中初始化 Conda 并启动：

```bash
source /data/xjc/miniconda3/etc/profile.d/conda.sh
conda activate 你的环境
python -u train.py --config configs/experiment.yaml
```

需要在某一行失败时立即终止整个多行任务，可以加上 `set -e`：

```bash
set -e
source /data/xjc/miniconda3/etc/profile.d/conda.sh
conda activate 你的环境
python -u train.py --seed 1
python -u train.py --seed 2
```

命令开头的 `CUDA_VISIBLE_DEVICES=0 python ...` 和 `export CUDA_VISIBLE_DEVICES=0` 换行写法会自动转成任务的物理 GPU 选择。与选择器冲突时会提示修改。其余位置的 GPU 赋值请移到选择器中。

GPU 通过 `CUDA_VISIBLE_DEVICES` 的 UUID 分配。分到物理 GPU 3 时，训练代码中通常仍使用 `cuda:0`；分到物理 GPU 3、5 时，对应进程内的 `cuda:0`、`cuda:1`。`cpu` 模式将该变量设为空字符串。

## 让 Codex / Claude Code 添加任务

可以。Codex、Claude Code 或其他能执行终端命令的工具，都可通过 `queue_cli.py` 把训练命令添加到**同一条队列**，网页会自动显示。它们需要能访问这台服务器上的客户端和运行中的队列服务。先启动 `app.py`，然后让工具调用：

```bash
python3 /data/xjc/train-queue/queue_cli.py add \
  --cwd /data/xjc/Flow_Matching_Same \
  --gpu 0 \
  --command 'python -u train.py --config configs/experiment.yaml'
```

返回 JSON，如 `{"id": 1}`，表示任务已入队。`--gpu auto` 自动等一张空闲 GPU；`--gpu 0,1` 等待指定两张卡。使用训练环境的 Python 绝对路径，或在任务里初始化 Conda。

对于多行或包含复杂引号的命令，可以先写 UTF-8 文本文件，再提交，避免 Shell 改写命令内容：

```bash
python3 /data/xjc/train-queue/queue_cli.py add \
  --cwd /data/xjc/Flow_Matching_Same \
  --gpu auto \
  --file /path/to/training-command.txt
```

`--file -` 表示从标准输入读取。文件里的全部内容作为**一个任务**，不会按行拆成多条任务。

同一入口也支持查看与管理：

```bash
python3 /data/xjc/train-queue/queue_cli.py status
python3 /data/xjc/train-queue/queue_cli.py log 1
python3 /data/xjc/train-queue/queue_cli.py cancel 1
python3 /data/xjc/train-queue/queue_cli.py retry 1
python3 /data/xjc/train-queue/queue_cli.py top 1
python3 /data/xjc/train-queue/queue_cli.py pause
python3 /data/xjc/train-queue/queue_cli.py resume
```

自定义端口时，将 `--port 8877` 放在 `add`、`status` 等子命令前。客户端自动获取页面会话令牌，不需要手动复制，且仅连接 `127.0.0.1`，不使用 HTTP 代理。请求失败以非零退出码和标准错误中的 JSON 报错；不会自动重试提交。遇到超时等结果不确定的情况，先查看 `status` 确认任务是否已入队，避免重复添加。

你可以直接对 Codex 或 Claude Code 说：

> 请把当前项目 seed=1、2、3 的三次训练依次加入 Train Queue，使用 GPU 0，提交后列出任务编号。通过 /data/xjc/train-queue/queue_cli.py 提交，训练启动命令使用项目的实际配置和 Python 环境。

前提是该会话能访问服务器上的客户端和队列服务；若当前会话的终端或本机网络操作需要审批，仍按会话权限执行。仅有一个网页聊天窗口、无法访问这台服务器的会话，不能直接调用这个本机入口。多个工具可以向同一服务提交，任务按服务收到请求的顺序入队。

### 自动找到队列入口

当前服务器已配置以下发现说明：

- `/data/xjc/AGENTS.md`：从 `/data/xjc` 作为工作区启动的 Codex 可读取。
- `/data/xjc/codex-home/AGENTS.md`：当前 `CODEX_HOME=/data/xjc/codex-home` 配置的全局说明，覆盖 `/data/xjc` 下各独立训练仓库的队列任务。
- `/data/xjc/CLAUDE.md`：Claude Code 从 `/data/xjc` 或其子目录启动时可沿父目录读取。
- `/data/xjc/train-queue/AGENT_USAGE.md`：两者共用的操作指南，说明服务检查、命令提交、环境选择和任务编号确认。

新会话在加载对应说明后，你可以直接说“把这三次训练加入队列，使用 GPU 0”。已经打开的会话可先发送：

> 请先阅读 /data/xjc/train-queue/AGENT_USAGE.md，然后通过 Train Queue 提交训练任务。

Codex 的项目说明搜索从当前项目根目录开始，不能只依赖项目根目录以外的父级 `AGENTS.md`。若另一个 Codex 会话使用了不同的 `CODEX_HOME`，将上面这句入口说明添加到它使用的全局 `AGENTS.md` 或当前训练仓库的 `AGENTS.md`。Claude Code 若配置了说明文件排除规则，也需确认 `/data/xjc/CLAUDE.md` 没有被排除。文件发现说明不会替代会话自身的终端与本机网络权限。

规则参考：[Codex AGENTS.md 官方说明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)、[Claude Code 项目记忆官方说明](https://code.claude.com/docs/en/memory)。

## 查看和管理之前启动的训练

页面的“服务器已有训练”会自动扫描**与服务同用户、同进程可见空间**的已有进程，不需要重新启动训练，也不要求训练最初由本工具提交。

- 默认显示当前用户的 GPU 计算进程、常见 Python 训练入口，以及 torchrun / accelerate / deepspeed 等启动的训练。分布式子进程和 DataLoader 子进程尽量按父子关系合并成一个任务。
- 非标准启动方式可能无法自动识别。切换“全部用户进程”，找到目标后点击“纳入监控”，即可加入默认训练列表。
- 每个任务显示启动命令、项目目录、运行时长、进程数量、CPU 占用、RSS 内存合计、GPU 及该组进程的显存用量。CPU 多核合计可以超过 100%，RSS 合计可能包含共享内存的重复统计。
- 自动寻找根进程和子进程的 stdout / stderr 指向的普通日志文件，例如 Shell 重定向或 nohup 输出。没有找到时，可点击“关联日志”，填写已有日志的绝对路径；该关联会持久保存。
- 支持常见文本和 JSON 行日志，解析 epoch / step / iteration、loss、lr 等。只有日志明确给出总量，或启动参数明确指定 `--steps` / `--max-steps` / `--epochs` 等总量时，才显示百分比；这是相应 step 或 epoch 的进度，不代表多阶段训练全部工作已完成。总量仅在 YAML 等配置文件里时，当前版本显示已到达的步数。
- “暂停进程”使用 SIGSTOP，“继续运行”使用 SIGCONT。**暂停不会释放 GPU 显存**，长时间暂停分布式任务可能触发通信超时。
- “停止”只发送 SIGTERM；程序是否保存 checkpoint 取决于训练代码。任务仍未退出时可以选择“强制结束”（SIGKILL）。已有训练不会自动从普通停止升级为强杀。
- 操作按当前用户的进程树逐个发送信号，不向整个共享 Shell 进程组广播。服务本身、其父进程和工具自有队列执行器不作为外部任务控制。队列提交的任务仍从队列列表管理。
- 进程身份包含 PID、启动时间和系统启动标识；发送信号时使用 pidfd 并再次核对用户与身份，避免 PID 复用导致误操作。Linux x86_64 / aarch64 的较旧 Conda Python 构建通过标准库 ctypes 调用同一内核接口。内核或会话权限不支持时保留查看功能。

输出直接在终端或管道中、且没有落盘时，无法恢复过去的训练日志；监控器不会读取管道或终端，也不会抢走原终端的输出。不显示猜测出来的训练完成百分比。外部进程结束后会出现在“本次监控中已结束”，但服务无法追溯其退出码；这部分历史保存在本次服务内存中。

Codex / Claude Code 也可直接使用：

```bash
python3 /data/xjc/train-queue/queue_cli.py processes
python3 /data/xjc/train-queue/queue_cli.py processes --all
python3 /data/xjc/train-queue/queue_cli.py process-log 完整进程身份
python3 /data/xjc/train-queue/queue_cli.py bind-log 完整进程身份 /data/xjc/项目/train.log
python3 /data/xjc/train-queue/queue_cli.py process-control 完整进程身份 pause
python3 /data/xjc/train-queue/queue_cli.py process-control 完整进程身份 resume
python3 /data/xjc/train-queue/queue_cli.py process-control 完整进程身份 stop
python3 /data/xjc/train-queue/queue_cli.py process-control 完整进程身份 kill
```

完整进程身份使用 `processes` 输出的 `identity`，不是只填写 PID。服务未启动时，`python3 /data/xjc/train-queue/process_monitor.py` 可直接做只读扫描诊断。

服务必须在训练服务器宿主环境中运行；如果启动在隔离的 Codex 沙箱或 Docker 进程命名空间中，它只能看到该命名空间里的进程。使用你的服务器终端启动本工具，再通过 SSH 转发打开网页即可管理宿主机上已有的训练。

技术依据：[Linux /proc 官方文档](https://www.kernel.org/doc/html/latest/filesystems/proc.html)、[Python pidfd 信号接口](https://docs.python.org/3/library/signal.html#signal.pidfd_send_signal)。

## 调度规则

- **严格顺序，一次一个任务**。队首等待 GPU 时，后面的任务也等待。需要优先执行另一项时可以“置顶”。
- GPU 必须同时满足：没有 `nvidia-smi` 报告的计算进程、已用显存不超过 512 MiB、利用率不超过 10%、空闲显存达到任务的最低要求。默认至少 1024 MiB 空闲，可在提交时修改。
- 检测失败或 GPU 不存在时，任务留在队列中并显示原因；不会把 GPU 任务自动当成 CPU 任务执行。
- 多 GPU 任务会等待指定的所有卡同时空闲。`auto` 当前只自动分配单卡。
- “暂停调度”仅暂停后续启动，当前训练继续。默认失败后继续下一项，可开启“任务失败后暂停队列”。
- 取消运行中任务时先发送 SIGTERM，最多等待 5 秒，再发送 SIGKILL，清理同一进程组的子进程。
- “重新排队”创建一条新任务，保留原任务与日志；不代表从模型 checkpoint 自动恢复。
- 每条任务都是独立的 `bash -lc`，前一任务中的 `cd`、`export`、Conda 激活不会传给下一条任务。请在每条任务中设置所需环境，或使用绝对路径。

提交的命令应在**前台运行直到训练结束**。请直接用 `python ...`、`torchrun ...`、`accelerate launch ...` 等启动，避免 `nohup`、`&`、`tmux`、`setsid` 等后台或另建会话的启动方式。任务结束时，执行器会清理其进程组内遗留的后台进程；主动另建会话的进程不属于这一管理范围。多行任务以 Bash 最终退出码判断成功与失败。

本工具管理自身队列并检查服务器当时的 GPU 状态。其他终端或其他用户仍能独立启动 GPU 程序；它不会为其他程序强制预留或限制显卡，也不会主动终止已有训练。

## 数据与重启

默认数据保存在项目的 `.data/` 中：

```text
.data/
  queue.sqlite3              # 任务、队列顺序、设置
  queue.lock                 # 防止同一数据目录运行多个调度器
  tasks/<任务编号>/
    output.log               # 完整训练输出
    supervisor.log           # 执行器自身输出
    spec.json                # 命令、工作目录和 GPU 环境
    process.json             # 训练进程身份
    release.json             # 持久化启动确认
    result.json              # 完成后的结果
```

关闭网页不影响调度。停止 Python 网页服务后，**已经启动的训练仍由独立执行器继续运行**；等待中的任务暂时停止调度。使用原数据目录重启服务，会接管运行中的任务，读取已完成任务的结果，然后继续队列。

若执行器异常消失，任务会标记“意外中断”，队列自动暂停；检查日志和训练进程后再恢复。程序不会自动重跑中断任务。服务器整机重启会使训练中断，需要自行从 checkpoint 恢复。

页面每秒更新任务和日志；为避免大日志拖慢网页，日志面板只保留最近 256 KiB。`output.log` 保存完整输出且不自动轮转，长期运行请留意磁盘空间。

本工具供本机用户使用，提交的命令以启动服务的系统用户权限执行。API 检查本机 Host、请求 Origin 和页面会话令牌，并拒绝跨站请求。它没有多用户账号体系，不要用反向代理直接暴露到公网。同机同账户进程不属于隔离边界。

## 验证

先选 `cpu`，依次提交两条短命令，即可验证界面与队列，无需占用 GPU：

```bash
python3 -u -c "import time; print('第一个任务开始'); time.sleep(5); print('第一个任务结束')"
```

```bash
python3 -u -c "print('第二个任务开始')"
```

运行自动测试：

```bash
cd /data/xjc/train-queue
python3 -m unittest discover -s tests -v
node --check static/app.js
```

测试在临时目录中启动短进程，覆盖顺序执行、GPU 不可用、多卡等待与环境注入、失败暂停、取消及强制终止、重启接管、异常恢复、日志和 HTTP 保护。GPU 部分使用模拟探针，不会启动真实训练。HTTP 测试需允许绑定本地临时端口。Node 仅用于 JavaScript 语法检查，运行工具本身不需要 Node。
