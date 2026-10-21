# tmux 轻量 GPU 队列

每组卡一个 tmux 窗口，实验依次执行；从其他终端随时追加。训练输出直接显示在执行窗口，没有网页或后台服务。

需要 Linux、Python 3.10+、NVIDIA 驱动和 `nvidia-smi`。工具只使用 Python 标准库。

## 在 ink_pde 中安装（一次即可）

激活训练环境，以可编辑模式安装：

```bash
conda activate ink_pde
python -m pip install -e /home/lsc/project/claude_code/train-queue
tq --help
```

以后新开终端只需 `conda activate ink_pde`，即可直接使用 `tq`，无需定义 Shell 函数或填写脚本路径。可编辑安装直接引用项目源码，修改代码后无需重新安装；移动项目目录或修改安装入口时重新执行安装命令。

如果当前终端曾定义过旧的 `tq` 函数，可执行 `unset -f tq`，让 Shell 使用安装后的命令。

## 开始一组实验

在 tmux 执行窗口中激活训练环境、进入研究项目：

```bash
conda activate ink_pde
cd /home/lsc/project/claude_code/inked_characters_recovery
tq work pair-a --gpus 2,3 --monitor-gpu 0
```

卡号仅为示例。执行器会一直等待任务；脱离 tmux 不影响执行，但不要关闭执行器所在的窗口。

在另一个终端激活同一环境，追加原来的训练命令：

```bash
conda activate ink_pde
tq add pair-a --command \
  'EXP_NAME=experiment_a bash tasks/rectified_flow/train_rain100h_noise_cond.sh'

tq add pair-a --command \
  'EXP_NAME=experiment_b bash tasks/rectified_flow/train_rain100h_noise_cond.sh'

tq list pair-a
```

任务使用 **work 执行窗口启动时的环境**。追加终端也可以直接使用 `/home/lsc/anaconda3/envs/ink_pde/bin/tq`，无需激活环境。换训练环境时，在目标环境中安装本工具并重启执行器。

任务默认在该组配置的工作目录运行，追加终端的当前目录不影响它；可通过 `add --cwd /项目目录` 单独覆盖。每条命令是一段完整 Bash 脚本，支持变量前缀、循环、多行内容。复杂命令也可用 `--command "$(cat command.txt)"` 传入。

任务在非登录 Bash 中执行；前一任务的 `cd`、`export` 不传给下一任务。多行命令需要遇错即停时，自行加 `set -e`。

### 用卡与并行

执行器传入 `GPU_IDS=2,3` 和 `MONITOR_GPU_ID=0`。研究项目现有脚本继续选择单卡 / torchrun 并设置 `CUDA_VISIBLE_DEVICES`；队列不改写这个变量。

**命令中省略 `GPU_IDS` 和 `MONITOR_GPU_ID`，采用所属队列的设置。** 当前研究工程使用 `${GPU_IDS:-默认值}` 等写法，可以继承。若新脚本强制改写卡号，需要先修正脚本或使用匹配的队列；工具不静态分析 Shell，也不提供硬件隔离。

同组按追加顺序执行，训练及自动评估全部退出后才启动下一项。普通非零退出码记录为 `failed` 并继续，不自动重试；信号退出及退出码 130/143 视为中断，需要检查后恢复。使用前台训练命令，避免 `nohup`、`&`、额外 tmux 窗口等将实验移出管理范围。

在另一个 tmux 窗口运行另一组，即可并行：

```bash
conda activate ink_pde
cd /home/lsc/project/claude_code/inked_characters_recovery
tq work pair-b --gpus 4,5 --monitor-gpu 1
```

支持一组四卡、两组双卡或四组单卡。每组需要自己的独立评估卡，不能与该组训练卡重叠。不同组若配置了相同的卡，会等待彼此释放文件锁，不能同时运行。

队列为空时不持有 GPU 锁。有待运行或运行任务时不能更改该组卡号，请使用新队列名。

## 空闲与显式共享

默认要求每张训练卡和评估卡：没有计算进程、已用显存 ≤512 MiB、利用率 ≤10%。每隔 2 秒检测，连续两次满足后取得全部 GPU 锁，并在启动前再次检查。

其他用户的暂停训练若长期占用少量显存，可显式共享：

```bash
tq work pair-a --gpus 2,3 --monitor-gpu 0 \
  --share --train-free-gib 18 --monitor-free-gib 6
```

示例表示**每张**训练卡至少空闲 18 GiB，评估卡至少空闲 6 GiB。门槛应包含你的实验峰值和预留余量，工具不自动估算需求。共享模式不因已有进程或利用率阻止启动。

配置会保存，之后 `tq work pair-a` 沿用；`tq work pair-a --no-share` 切回默认模式。特殊任务可覆盖门槛：

```bash
tq add pair-a --train-free-gib 20 --monitor-free-gib 8 --command \
  'EXP_NAME=large_experiment bash tasks/rectified_flow/train_rain100h_noise_cond.sh'
```

多卡逐张检查，不合计显存。显卡缺失、检测失败、锁冲突时等待并显示原因。文件锁保护使用**同一数据目录**的执行器，其他用户或直接启动的程序不受锁约束；共享只是启动前检查，外部训练恢复后仍可能继续申请显存。

## 查看、取消与中断

```bash
tq list             # 所有队列
tq list pair-a      # 某一组
tq cancel 12        # 仅取消待运行任务 #12
```

在执行窗口按 Ctrl+C，中断当前命令并等待其收尾，然后退出执行器，保留后续任务。工具保留原控制终端与进程组，不另建训练会话；研究工程的 supervisor 仍负责清理它自己的训练子进程。不增加强杀或跨会话清理。

等待时按 Ctrl+C，任务保持 `queued`，可直接 `tq work pair-a` 继续。

运行中中断或执行器异常退出后，先检查再恢复：

```bash
tq list pair-a
# 检查终端、nvidia-smi 和训练进程，确认旧任务不再运行
tq work pair-a --resume
```

中断项不会重跑，后续任务保留。已记录的命令进程仍存活时拒绝恢复，身份通过 PID、启动时钟和系统启动标识核对。命令进程退出不代表另建会话的后代已退出，仍需人工检查。重跑或从 checkpoint 恢复需重新添加原训练命令。

## 数据位置

默认 `${XDG_STATE_HOME:-$HOME/.local/state}/gpu-queue-lite/`，当前账户通常是：

```text
/home/lsc/.local/state/gpu-queue-lite/
├── queue.sqlite3       # 配置、命令、目录、顺序、状态、时间、退出码
├── queue.sqlite3-wal   # 使用时可能产生的 SQLite 辅助文件
├── queue.sqlite3-shm
└── locks/              # 锁文件残留不代表锁仍被持有
```

首次调用创建，仅当前用户可访问。SQLite 是本地文件，无需数据库服务。训练输出直接显示在 tmux，原项目自行保存日志和模型；队列不向研究工程写入数据。

自定义目录放在子命令前；所有终端须使用同一目录才能共享队列和锁：

```bash
tq --data-dir /tmp/tq-test list
```

## 验证与旧版

```bash
python -m unittest discover -s tests -v
```

测试在临时目录使用模拟 nvidia-smi 和短命令，不启动真实训练，也不修改默认队列。

旧版及两个参考脚本保存在 Git 分支 `legacy/web-queue`：

```bash
git show legacy/web-queue:README.md
git show 'legacy/web-queue:bash/wait2R(1).sh'
```

旧版无运行数据需要迁移。新入口不使用旧数据库、网页、API 或执行器。
