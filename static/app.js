const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="queue-token"]').content;
const labels = {queued:'等待执行',starting:'启动中',running:'运行中',canceling:'正在取消',succeeded:'已完成',failed:'失败',canceled:'已取消',interrupted:'意外中断'};
const terminal = new Set(['succeeded','failed','canceled','interrupted']);
let state = null, filter = 'all', initialized = false, taskSignature = '', gpuSignature = '';
let selectedLog = null, logOffset = 0, logText = '', toastTimeout, polling = false;
let externalState = null, selectedProcessLog = null, bindIdentity = null, logFileId = null;
const processLabels = {running:'运行中',paused:'已暂停',stopping:'正在停止',exited:'已结束'};
const escapeHTML = str => String(str ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const date = timestamp => new Date(timestamp*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});

async function api(path, body) {
  const options = {headers:{'X-Queue-Token':token}, cache:'no-store'};
  if (body !== undefined) Object.assign(options, {method:'POST',headers:{...options.headers,'Content-Type':'application/json'},body:JSON.stringify(body)});
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}
function toast(message, error = false) {
  $('toast').textContent = message;
  $('toast').classList.toggle('error', error);
  $('toast').hidden = false;
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => $('toast').hidden = true, 4000);
}
function persistForm() {
  try { localStorage.setItem('train-queue-draft',JSON.stringify({command:$('command').value,cwd:$('cwd').value,gpu:$('gpu').value,memory:$('memory').value})); } catch (_) {}
}
function renderGPUs() {
  const signature = JSON.stringify([state.gpus,state.gpu_error]);
  if (signature === gpuSignature) return;
  gpuSignature = signature;
  $('gpu-list').innerHTML = state.gpus.length ? state.gpus.map(g => `<div class="gpu-card"><div class="gpu-top"><strong>GPU ${escapeHTML(g.index)}</strong><span class="${g.busy || g.used_mb>512 || g.utilization>10?'busy':''}">${g.busy || g.used_mb>512 || g.utilization>10?'使用中':'空闲'}</span></div><div class="gpu-name">${escapeHTML(g.name)}</div><div class="gpu-bar"><div style="width:${Math.min(100,Math.max(0,g.used_mb/g.total_mb*100))}%"></div></div><div class="gpu-memory">${(g.used_mb/1024).toFixed(1)} / ${(g.total_mb/1024).toFixed(1)} GiB · ${g.utilization}%</div></div>`).join('') : '<div class="gpu-unavailable"><strong>GPU 暂不可用</strong>训练任务会保持等待。可以选择 cpu 验证排队流程。</div>';
  $('gpu-options').innerHTML = '<option value="auto">自动选择空闲 GPU</option><option value="cpu">CPU / 调试</option>' + state.gpus.map(g=>`<option value="${escapeHTML(g.index)}">GPU ${escapeHTML(g.index)} · ${escapeHTML(g.name)}</option>`).join('');
}
function renderTasks() {
  const signature = JSON.stringify([state.tasks,filter]);
  if(signature === taskSignature) return;
  taskSignature = signature;
  const tasks = state.tasks.filter(t => filter === 'all' || (filter === 'active' ? !terminal.has(t.status) : terminal.has(t.status)));
  if (!tasks.length) {
    $('tasks').innerHTML = `<div class="empty"><div class="empty-icon">›_</div><strong>${filter === 'history'?'还没有已结束的任务':filter === 'active'?'目前没有进行中的任务':'队列准备好了'}</strong><p>${filter === 'all'?'输入第一条训练命令，按 Enter 加入队列。':'任务状态会自动更新。'}</p></div>`;
    return;
  }
  // Keep the pending/running queue in execution order; show newest history first.
  tasks.sort((a,b) => {
    const ah=terminal.has(a.status), bh=terminal.has(b.status);
    if (ah !== bh) return ah ? 1 : -1;
    if (ah) return b.id-a.id;
    const ar=['running','canceling'].includes(a.status),br=['running','canceling'].includes(b.status);
    if(ar!==br) return ar?-1:1;
    return a.position-b.position || a.id-b.id;
  });
  $('tasks').innerHTML = tasks.map(t => {
    const gpu = t.assigned.length ? `GPU ${t.assigned.join(', ')}` : t.gpu === 'auto'?'GPU 自动分配':t.gpu === 'cpu'?'CPU / 调试':`GPU ${t.gpu}`;
    const timeInfo = t.finished_at ? `${date(t.created_at)} · 耗时 ${duration(t.finished_at-(t.started_at||t.created_at))}` : `${date(t.created_at)} 提交`;
    return `<article class="task" data-id="${t.id}"><div class="task-head"><div class="task-title"><span class="task-id">#${String(t.id).padStart(3,'0')}</span><span class="badge ${t.status}">${labels[t.status] || t.status}</span></div><span class="task-time">${timeInfo}</span></div><pre class="task-command">${escapeHTML(t.command)}</pre><div class="task-bottom"><div class="task-info"><span class="task-cwd" title="${escapeHTML(t.cwd)}">⌂ ${escapeHTML(t.cwd)}</span><span>▧ ${escapeHTML(gpu)}</span>${t.exit_code !== null?`<span>退出码 ${t.exit_code}</span>`:''}</div><div class="task-actions"><button class="text-button" data-action="log">日志</button>${t.status==='queued'?'<button class="text-button" data-action="top">置顶</button>':''}${terminal.has(t.status)?'<button class="text-button" data-action="retry">重新排队</button>':t.status!=='canceling'?'<button class="text-button cancel" data-action="cancel">取消</button>':''}</div></div>${t.note?`<div class="task-note">${escapeHTML(t.note)}</div>`:''}</article>`;
  }).join('');
}
function duration(seconds) {
  seconds = Math.max(0, Math.floor(seconds));
  return seconds < 60 ? `${seconds} 秒` : seconds < 3600 ? `${Math.floor(seconds/60)} 分 ${seconds%60} 秒` : `${Math.floor(seconds/3600)} 小时 ${Math.floor(seconds%3600/60)} 分`;
}
function renderExternal() {
  if(!externalState) return;
  const mode = $('external-mode').value;
  const tasks = mode==='history'?externalState.history:externalState.processes;
  const liveTraining=externalState.processes.filter(t=>t.is_training);
  const queueRunning=state?.tasks.filter(t=>['starting','running','canceling'].includes(t.status)).length || 0;
  $('running-count').textContent=queueRunning+liveTraining.filter(t=>['running','stopping'].includes(t.status)).length;
  $('total-count').textContent=(state?.tasks.length || 0)+liveTraining.length;
  $('external-count').textContent=tasks.length;
  $('external-description').textContent=`当前用户 ${externalState.username} · 每两秒扫描 · 分布式子进程合并显示`;
  const notices=[externalState.gpu_error,!externalState.control_available?'当前内核或会话权限不支持安全进程控制，仍可查看状态和日志。':null].filter(Boolean);
  $('external-notice').hidden=!notices.length;
  $('external-notice').textContent=notices.join(' ');
  if(!tasks.length) {
    $('external-tasks').innerHTML=`<div class="empty"><strong>${mode==='history'?'本次监控中暂无已结束任务':'未发现可显示的进程'}</strong><p>${mode==='training'?'可切换“全部用户进程”，手动将未识别的训练纳入监控。':'进程需要与服务位于同一服务器、同一可见进程空间。'}</p></div>`;
    return;
  }
  $('external-tasks').innerHTML=tasks.map(t=>{
    const progress=t.progress || {}, metrics=Object.entries(progress.metrics || {}).slice(0,5);
    const gpuGroups=new Map();
    for(const gpu of t.gpus) {
      const value=gpuGroups.get(gpu.index) || {memory:0,known:false};
      if(gpu.memory_mb!==null) {value.memory+=gpu.memory_mb;value.known=true;}
      gpuGroups.set(gpu.index,value);
    }
    const gpuText=[...gpuGroups].map(([index,value])=>`GPU ${index}${value.known?' · '+(value.memory/1024).toFixed(1)+' GiB':''}`).join(' / ');
    const progressText=progress.current!==null?`${progress.unit} ${progress.current}${progress.total!==null?' / '+progress.total:''}`:'尚无可识别的训练进度';
    const ended=t.status==='exited', canControl=externalState.control_available && !ended;
    const actions=`<button class="text-button" data-process-action="log">日志</button>${!ended?'<button class="text-button" data-process-action="bind">关联日志</button>':''}${!t.is_training&&!ended?'<button class="text-button" data-process-action="track">纳入监控</button>':''}${canControl?`<button class="text-button" data-process-action="${t.status==='paused'?'resume':'pause'}">${t.status==='paused'?'继续运行':'暂停进程'}</button><button class="text-button cancel" data-process-action="${t.status==='stopping'?'kill':'stop'}">${t.status==='stopping'?'强制结束':'停止'}</button>`:''}`;
    return `<article class="task external-task" data-identity="${escapeHTML(t.identity)}"><div class="task-head"><div class="task-title"><span class="task-id">PID ${t.pid}</span><span class="badge ${t.status==='paused'?'canceling':t.status==='stopping'?'failed':t.status==='exited'?'succeeded':'running'}">${processLabels[t.status]}</span><span class="external-origin">已有进程 · ${t.members.length} 个进程</span></div><span class="task-time">${ended?'运行':'已运行'} ${duration(t.elapsed_seconds)}</span></div><pre class="task-command">${escapeHTML(t.command)}</pre>${t.is_training?`<div class="training-progress"><span>${escapeHTML(progressText)}</span><span>${progress.percent!==null?progress.percent+'%':''}</span></div>${progress.percent!==null?`<div class="progress-track"><div style="width:${progress.percent}%"></div></div>`:''}<div class="training-metrics">${metrics.map(([key,value])=>`<span>${escapeHTML(key)} <b>${escapeHTML(value)}</b></span>`).join('')}</div>`:''}<div class="task-bottom"><div class="task-info"><span class="task-cwd" title="${escapeHTML(t.cwd)}">⌂ ${escapeHTML(t.cwd || '目录不可读')}</span><span>CPU ${t.cpu_percent===null?'采样中':t.cpu_percent+'%'}</span><span>内存 ${(t.rss_mb/1024).toFixed(2)} GiB</span>${gpuText?`<span>${escapeHTML(gpuText)}</span>`:''}</div><div class="task-actions">${actions}</div></div>${t.log_path?`<div class="external-log-path" title="${escapeHTML(t.log_path)}">日志：${escapeHTML(t.log_path)}${t.log_modified_at?' · '+date(t.log_modified_at)+' 更新':''}</div>`:t.is_training?'<div class="task-note">未找到日志文件；可关联已有日志。终端或管道中的历史输出无法直接补读。</div>':''}${t.note?`<div class="task-note">${escapeHTML(t.note)}</div>`:''}${t.log_error?`<div class="task-note">${escapeHTML(t.log_error)}</div>`:''}</article>`;
  }).join('');
}
function render() {
  const counts = state.tasks.reduce((a,t)=>(a[t.status]=(a[t.status]||0)+1,a),{});
  $('running-count').textContent = (counts.starting||0)+(counts.running||0)+(counts.canceling||0);
  $('queued-count').textContent = counts.queued||0;
  $('done-count').textContent = counts.succeeded||0;
  $('failed-count').textContent = (counts.failed||0)+(counts.interrupted||0);
  $('total-count').textContent = $('tab-count').textContent = state.tasks.length;
  $('pause').textContent = state.settings.paused?'恢复调度':'暂停调度';
  $('stop-on-failure').checked = state.settings.stop_on_failure;
  const notices = [];
  if(state.settings.paused) notices.push('队列已暂停。正在运行的任务继续执行，后续任务等待恢复调度。');
  if(state.scheduler_error) notices.push('调度器异常：'+state.scheduler_error);
  if(state.gpu_error) notices.push(state.gpu_error);
  $('notice').hidden = !notices.length;
  $('notice').textContent = notices.join(' ');
  renderGPUs(); renderTasks();
}
async function refresh() {
  state = await api('/api/state');
  if (!initialized) {
    initialized = true;
    $('cwd').value = state.default_cwd;
    try {
      const draft = JSON.parse(localStorage.getItem('train-queue-draft') || '{}');
      for (const id of ['command','cwd','gpu','memory']) if(typeof draft[id] === 'string') $(id).value = draft[id];
    } catch (_) {}
  }
  $('connection').textContent = '● 已连接';
  $('connection').classList.remove('offline');
  render();
  try {
    const mode=$('external-mode').value==='all'?'all':'training';
    externalState=await api('/api/processes?mode='+mode);
    renderExternal();
  } catch(error) {
    $('external-notice').hidden=false;
    $('external-notice').textContent='进程监控暂不可用：'+error.message;
  }
}
async function refreshLog() {
  const taskId = selectedLog;
  const processId = selectedProcessLog, offset=logOffset;
  if (!taskId && !processId) return;
  const result = await api(processId?`/api/processes/${processId}/log?offset=${offset}`:`/api/tasks/${taskId}/log?offset=${offset}`);
  if (selectedLog !== taskId || selectedProcessLog !== processId || logOffset!==offset) return;
  if(processId && logFileId && result.file_id && result.file_id!==logFileId) {
    logText='[日志文件已轮换]\n'; logOffset=0; logFileId=result.file_id;
    await refreshLog(); return;
  }
  if(processId) logFileId=result.file_id;
  logOffset = result.offset;
  if(result.truncated) logText = '[仅显示最近的日志]\n';
  logText += result.text;
  if(logText.length>262144) logText='[仅显示最近的日志]\n'+logText.slice(-262144);
  $('log-output').textContent = logText || result.note || '暂无输出，等待任务启动或程序输出日志…';
  const task = state?.tasks.find(t=>t.id===taskId);
  if(task) $('log-meta').textContent = `${labels[task.status]} · ${task.cwd}`;
  if(processId) $('log-meta').textContent=result.path || '已有进程 · 只读取普通文件日志';
  if($('auto-scroll').checked) $('log-output').scrollTop = $('log-output').scrollHeight;
}
async function poll() {
  if(polling) return;
  polling=true;
  try {
    await refresh();
    try {await refreshLog();} catch(error) {$('log-meta').textContent='日志读取失败：'+error.message;}
  }
  catch(error) {
    $('connection').textContent = '● 连接中断';
    $('connection').classList.add('offline');
    $('notice').hidden = false;
    $('notice').textContent = `连接失败：${error.message}。正在自动重连。`;
  } finally { polling=false; }
}
$('submit-form').addEventListener('submit',async event=>{
  event.preventDefault();
  if(!initialized || $('submit').disabled) return;
  const command=$('command').value.trim();
  if(!command) return;
  $('submit').disabled = true;
  $('submit-error').hidden = true;
  try {
    const result = await api('/api/tasks',{command,cwd:$('cwd').value,gpu:$('gpu').value,min_free_mb:Number($('memory').value)});
    $('command').value=''; persistForm();
    toast(`任务 #${result.id} 已加入队列`);
    await refresh(); $('command').focus();
  } catch(error) { $('submit-error').hidden=false; $('submit-error').textContent=error.message; }
  finally { $('submit').disabled=false; }
});
$('command').addEventListener('keydown',event=>{
  if(event.key==='Enter' && !event.shiftKey && !event.isComposing && event.keyCode!==229) {
    event.preventDefault(); $('submit-form').requestSubmit();
  }
});
for(const id of ['command','cwd','gpu','memory']) $(id).addEventListener('input',persistForm);
$('pause').addEventListener('click',async()=>{
  if(!state) return;
  try { await api('/api/settings',{paused:!state.settings.paused}); await refresh(); }
  catch(error) {toast(error.message,true);}
});
$('stop-on-failure').addEventListener('change',async()=>{
  try {await api('/api/settings',{stop_on_failure:$('stop-on-failure').checked}); await refresh();}
  catch(error) {toast(error.message,true); if(state) $('stop-on-failure').checked=state.settings.stop_on_failure;}
});
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{
  filter=button.dataset.filter;
  document.querySelectorAll('[data-filter]').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-selected',String(b===button));});
  if(state) renderTasks();
}));
$('all-tasks').addEventListener('click',()=>{document.querySelector('[data-filter="all"]').click(); $('command').focus();});
$('tasks').addEventListener('click',async event=>{
  const button=event.target.closest('[data-action]');
  if(!button) return;
  const id=Number(button.closest('[data-id]').dataset.id), action=button.dataset.action;
  if(action==='log') {
    selectedLog=id; selectedProcessLog=null; logFileId=null; logOffset=0; logText=''; $('log-title').textContent=`任务 #${id} · 实时日志`;
    $('log-output').textContent='正在读取日志…'; $('log-dialog').showModal();
    try {await refreshLog();} catch(error) {$('log-output').textContent=error.message;}
    return;
  }
  button.disabled=true;
  try {
    const result=await api(`/api/tasks/${id}/${action}`,{});
    toast(action==='retry'?`已创建新任务 #${result.id}`:action==='top'?'任务已移至等待队列首位':'已提交取消请求');
    await refresh();
  } catch(error) {toast(error.message,true);button.disabled=false;}
});
$('close-log').addEventListener('click',()=>$('log-dialog').close());
$('log-dialog').addEventListener('close',()=>{selectedLog=null;selectedProcessLog=null;});
$('copy-log').addEventListener('click',async()=>{
  try {await navigator.clipboard.writeText(logText);toast('日志已复制');}
  catch (_) {toast('无法访问剪贴板，请在日志中手动选择复制',true);}
});
$('external-mode').addEventListener('change',()=>poll());
$('external-tasks').addEventListener('click',async event=>{
  const button=event.target.closest('[data-process-action]');
  if(!button) return;
  const identity=button.closest('[data-identity]').dataset.identity, action=button.dataset.processAction;
  const task=[...(externalState?.processes || []),...(externalState?.history || [])].find(t=>t.identity===identity);
  if(!task) return;
  if(action==='log') {
    selectedLog=null; selectedProcessLog=identity; logFileId=null; logOffset=0; logText='';
    $('log-title').textContent=`已有训练 PID ${task.pid} · 日志`;
    $('log-output').textContent='正在读取日志…'; $('log-dialog').showModal();
    try {await refreshLog();} catch(error) {$('log-output').textContent=error.message;}
    return;
  }
  if(action==='bind') {
    bindIdentity=identity; $('bind-path').value=task.log_path || ''; $('bind-error').hidden=true;
    $('bind-dialog').showModal(); return;
  }
  if(['stop','kill','pause'].includes(action)) {
    const message=action==='pause'?'暂停不会释放 GPU 显存；分布式通信可能因长时间暂停而超时。':action==='kill'?'强制结束不会给程序保存 checkpoint 的机会。':'发送终止信号，程序是否保存 checkpoint 取决于自身处理。';
    if(!window.confirm(`${action==='pause'?'暂停':action==='kill'?'强制结束':'停止'} PID ${task.pid} 及其当前用户子进程（共 ${task.members.length} 个）？\n${task.command}\n\n${message}`)) return;
  }
  button.disabled=true;
  try {
    await api(`/api/processes/${identity}/${action}`,{});
    toast(action==='track'?'已纳入训练监控':'操作已发送，正在更新进程状态'); await refresh();
  } catch(error) {toast(error.message,true);button.disabled=false;}
});
$('close-bind').addEventListener('click',()=>$('bind-dialog').close());
$('bind-form').addEventListener('submit',async event=>{
  event.preventDefault(); if(!bindIdentity) return;
  try {
    await api(`/api/processes/${bindIdentity}/bind-log`,{path:$('bind-path').value});
    $('bind-dialog').close(); toast('日志已关联'); await refresh();
  } catch(error) {$('bind-error').hidden=false;$('bind-error').textContent=error.message;}
});
poll(); setInterval(poll,1000);
