import './style.css';
import { EefScene } from './scene.js';
import { nearestFrame, nextAnomaly } from './timeline.js';
import { VideoGroup, selectCameraSlots } from './video-group.js';

const $ = selector => document.querySelector(selector);
const labels = { admitted: '可计算', pending_units: '单位待核实', excluded: '已排除', deferred: '暂缓', no_eef: '无有效 EEF' };
const esc = text => String(text ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
let stored = {}; try { stored = JSON.parse(localStorage.getItem('eef-review') || '{}'); } catch { /* invalid preferences */ }
$('#app').innerHTML = `
<header><div><h1><span>ASTRIBOT / RESEARCH</span>EEF 轨迹审查</h1><p>数据集视频 · 原始目标 · 运动学回放</p></div><div class="header-actions"><button id="toggle-catalog" aria-controls="episode-directory" aria-expanded="true">收起目录</button><div id="connection">连接共享服务…</div></div></header>
<div class="layout"><aside id="episode-directory"><label for="search">查找任务或数据集</label><input id="search" placeholder="搜索轨迹…"><label for="source">数据来源</label><select id="source" aria-label="数据来源"><option value="">全部来源</option></select><label for="group">映射类别</label><select id="group"><option value="">全部候选类别</option></select><select id="filter" aria-label="准入状态"><option value="">全部审查状态</option><option value="admitted">可计算</option><option value="pending_units">单位待核实</option><option value="excluded">已排除</option><option value="deferred">暂缓</option><option value="no_eef">无有效 EEF</option></select><select id="processing" aria-label="计算状态"><option value="">全部计算状态</option><option value="complete">已完成缓存</option><option value="anomaly">含异常帧</option><option value="queued">排队中</option><option value="running">计算中</option><option value="failed">计算失败</option><option value="uncomputed">尚未计算</option></select><div id="count" class="muted"></div><div id="catalog"></div><div id="pagination"><button id="prev-page">上一页</button><span id="page-number"></span><button id="next-page">下一页</button></div><div class="muted">源数据只读 · 含已审核的 Franka 数据<br>每个浏览器独立播放，共享计算缓存</div></aside>
<main><div class="titlebar"><div><h2 id="title">选择一条轨迹</h2><div id="subtitle" class="muted">从左侧目录开始，或等待恢复上次查看的位置。</div></div><div class="toolbar"><select id="mode" aria-label="机器人回放通道"><option value="state">State 回放</option><option value="action">Action 回放</option></select><label><input type="checkbox" id="action-layer">Action 目标</label></div></div>
<div id="notice">源 base 暂按 SDK chassis；跨本体基座尚未对齐。此页用于运动学审查。</div>
<div class="views"><section class="panel"><div class="panel-head"><span>ASTRIBOT S1 / 3D</span><label class="frame-toggle"><input id="reference-visible" type="checkbox" checked>参考系</label><span id="loading">载入共享模型…</span></div><div id="scene"></div><div class="legend"><span class="left">● 左臂目标</span><span class="right">● 右臂目标</span><span>○ 达成 EEF</span><span>— 位置残差</span></div><div id="frame-note" class="frame-note"></div></section>
<section class="panel video-panel"><div class="panel-head"><span>数据集视频 / 多视图同步</span><select id="camera" aria-label="主相机"></select></div><div class="camera-grid">
<div class="camera-card camera-head"><div class="camera-caption"><strong>主相机</strong><span id="head-status">等待轨迹</span></div><video id="video" preload="metadata" muted playsinline></video><div id="head-message" class="camera-message">等待轨迹</div></div>
<div class="camera-card camera-left"><div class="camera-caption"><strong>LEFT</strong><span id="left-status">等待轨迹</span></div><video id="video-left" preload="metadata" muted playsinline></video><div id="left-message" class="camera-message">等待轨迹</div></div>
<div class="camera-card camera-right"><div class="camera-caption"><strong>RIGHT</strong><span id="right-status">等待轨迹</span></div><video id="video-right" preload="metadata" muted playsinline></video><div id="right-message" class="camera-message">等待轨迹</div></div>
</div><div class="video-actions"><span id="video-kind">主相机驱动时间轴</span><button id="preview">生成 / 使用 H.264 预览</button></div><div id="media-status">可用相机共用时间轴</div></section></div>
<section class="panel timeline"><div class="controls"><button id="play" class="primary" disabled>▶ 播放</button><button id="previous" disabled>上一帧</button><button id="next" disabled>下一帧</button><button id="anomaly" disabled>下一异常</button><select id="speed" aria-label="播放速度"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><button id="compute" disabled>计算运动学回放</button><span id="time">0.000 s / 0 帧</span></div><input id="seek" type="range" min="0" max="0" value="0" step="1" disabled aria-label="轨迹帧"><div id="sync">视频实际呈现时间驱动机器人</div></section>
<div class="diagnostics"><section class="panel diagnostic"><h3>末端误差 / 当前位置 · 姿态</h3><div class="metric-row"><div><div class="muted left-text">左臂</div><div id="left-error" class="metric">—</div><div id="left-angle" class="muted">—</div></div><div><div class="muted right-text">右臂</div><div id="right-error" class="metric">—</div><div id="right-angle" class="muted">—</div></div></div></section><section class="panel diagnostic"><h3 id="gripper-title">夹爪 / 0 开 · 100 闭</h3><canvas id="gripper-plot"></canvas><div id="grippers" class="muted">—</div></section><section class="panel diagnostic"><h3>计算与诊断</h3><div id="diagnosis" class="status-details">尚未计算。目标和视频可先行检查。</div></section></div><div id="error" role="alert"></div><details><summary>来源、坐标与计算记录</summary><pre id="provenance"></pre></details><footer>残差取原始采样点。失败帧显示保留姿态并明确标记；暂停、拖动和倍速只读取缓存。</footer></main></div>`;

$('#toggle-catalog').onclick = () => {
  const collapsed = $('.layout').classList.toggle('catalog-collapsed');
  $('#toggle-catalog').textContent = collapsed ? '展开目录' : '收起目录';
  $('#toggle-catalog').setAttribute('aria-expanded', String(!collapsed));
};

const scene = new EefScene($('#scene'));
const video = $('#video');
let episode, trajectory, manifest, jobKey, index = 0, generation = 0, offset = 0, total = 0, selectedMode = stored.mode || 'state';
let controller = new AbortController(), chunks = new Map(), seekSerial = 0, seeking = false, syncSupported = true;
let presented = 0, model, catalogGeneration = 0;
const videos = new VideoGroup({head:video,left:$('#video-left'),right:$('#video-right')}, {
  onFrame(time, supported) {
    presented = time; syncSupported = supported;
    if (!seeking && !videos.seeking && trajectory) display(nearestFrame(trajectory.frames.map(f => f.sample_timestamp), time), time);
  },
  onStatus: updateMediaStatus,
});
$('#reference-visible').onchange = () => scene.setReferenceVisible($('#reference-visible').checked);
$('#mode').value = selectedMode; $('#speed').value = String(stored.speed || 1); $('#action-layer').checked = !!stored.actionLayer;

function showError(error) { if (error.name !== 'AbortError') $('#error').textContent = error.message || error; }
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
  if (!response.ok) throw new Error((await response.text()).slice(0, 600));
  return response.json();
}
function save() {
  if (episode) {
    stored = { episode: episode.id, camera: $('#camera').value,
      time: trajectory?.frames[index]?.sample_timestamp || 0, speed: +$('#speed').value, mode: selectedMode, actionLayer: $('#action-layer').checked };
    localStorage.setItem('eef-review', JSON.stringify(stored));
  }
}
function pause() { videos.pause(); $('#play').textContent = '▶ 播放'; }
function metricAllowed() { return episode?.review.status === 'admitted'; }
function plottedGrippers() {
  if (!trajectory) return;
  const canvas = $('#gripper-plot'), rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio; canvas.height = rect.height * devicePixelRatio;
  const ctx = canvas.getContext('2d'); ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.strokeStyle = '#344752'; ctx.beginPath(); ctx.moveTo(0, rect.height - 4); ctx.lineTo(rect.width, rect.height - 4); ctx.stroke();
  for (const [side, color] of [['left', '#5de0c0'], ['right', '#b19aff']]) {
    ctx.strokeStyle = color; ctx.beginPath(); let valid = false;
    trajectory.frames.forEach((frame, i) => {
      const value = frame[selectedMode].grippers[side];
      if (value == null) { valid = false; return; }
      const x = i / Math.max(1, trajectory.frames.length - 1) * rect.width, y = rect.height - 5 - value / 100 * (rect.height - 10);
      valid ? ctx.lineTo(x, y) : ctx.moveTo(x, y); valid = true;
    }); ctx.stroke();
  }
  ctx.strokeStyle = '#dae9ed'; ctx.beginPath(); const x = index / Math.max(1, trajectory.frames.length - 1) * rect.width;
  ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); ctx.stroke();
}
async function solvedFrame(i) {
  if (!manifest) return null;
  const key = jobKey, number = Math.floor(i / manifest.chunk_size), size = manifest.chunk_size;
  if (!chunks.has(number)) chunks.set(number, api(`/api/results/${key}/chunks/${number}`, { signal: controller.signal }).catch(error => { if (key === jobKey) chunks.delete(number); throw error; }));
  const data = await chunks.get(number);
  return data.frames[i % size];
}
function updateSync(frame, actualTime) {
  if (!frame || !episode) return;
  const delta = Math.abs(actualTime - frame.sample_timestamp);
  let bad = delta > 1 / (videos.head.camera?.fps || episode.fps);
  const parts = [`${syncSupported ? '实际呈现' : '降级时钟'} · 主相机偏差 ${(delta * 1000).toFixed(1)} ms`];
  for (const name of ['left', 'right']) {
    const slot = videos.slots[name];
    if (slot.state !== 'ready' || slot.presented == null) { parts.push(`${name.toUpperCase()} ${slot.state === 'missing' ? '无此相机' : '尚未同步'}`); continue; }
    const drift = Math.abs(slot.presented - actualTime), unsynced = drift > 1 / slot.camera.fps;
    parts.push(`${name.toUpperCase()} ${(drift * 1000).toFixed(1)} ms${unsynced ? '（未同步）' : ''}`); bad ||= unsynced;
  }
  $('#sync').textContent = parts.join(' · '); $('#sync').classList.toggle('error-state', bad);
}
function updateMediaStatus() {
  const messages = {empty:'等待轨迹',missing:'数据未提供此相机',loading:'读取视频…',preview:'准备 H.264 预览…',error:'视频未就绪'};
  for (const [name, slot] of Object.entries(videos.slots)) {
    const ready = slot.state === 'ready';
    const promoted = slot.state === 'missing' && videos.head.camera?.camera === `observation.images.${name}`;
    const text = slot.error || (promoted ? '该相机已用作主视图' : messages[slot.state]) || '';
    $(`#${name}-status`).textContent = ready ? `${slot.camera.kind === 'formal_episode_preview' ? '正式片段 H.264' : slot.derived ? '预览 H.264' : '样本 H.264'} · ${slot.camera.fps} FPS` : messages[slot.state];
    $(`#${name}-message`).hidden = ready && !slot.error;
    $(`#${name}-message`).textContent = text;
  }
  const ready = videos.ready.length, available = Object.values(videos.slots).filter(slot => slot.camera).length;
  $('#media-status').textContent = videos.head.buffering ? '主相机缓冲中，左右视图等待同步' : `${ready}/${available} 路视频就绪 · 主相机驱动机器人`;
  $('#play').textContent = videos.playIntent ? 'Ⅱ 暂停' : '▶ 播放';
  $('#play').disabled = !trajectory || !videos.allReady || videos.head.state !== 'ready';
  if (!seeking && trajectory) updateSync(trajectory.frames[index], presented);
}
async function display(i, actualTime = presented) {
  if (!trajectory) return;
  const token = generation; index = Math.max(0, Math.min(trajectory.frames.length - 1, i)); const expected = index;
  const frame = trajectory.frames[index]; let solved = null;
  try { solved = await solvedFrame(index); } catch (error) { showError(error); }
  if (token !== generation || expected !== index) return;
  scene.showFrame(frame, solved, selectedMode, $('#action-layer').checked, metricAllowed());
  $('#seek').value = index; $('#time').textContent = `${frame.sample_timestamp.toFixed(3)} s / ${index + 1} · ${trajectory.frames.length} 帧`;
  updateSync(frame, actualTime);
  for (const side of ['left', 'right']) {
    const err = solved?.errors?.[side];
    $(`#${side}-error`).innerHTML = err ? `${err.position_mm.toFixed(1)} <small>mm</small>` : '—';
    $(`#${side}-angle`).textContent = err ? `${err.angle_deg.toFixed(2)}°` : frame[selectedMode].targets[side] ? '尚无本帧有效解' : '缺失 / 无效姿态';
  }
  $('#grippers').textContent = ['left', 'right'].map((side, i) => `${i ? '右' : '左'} ${frame[selectedMode].grippers[side]?.toFixed(1) ?? '缺失'}`).join('   /   ');
  if (solved) $('#diagnosis').textContent = `${solved.ik_success ? 'IK 返回成功' : 'IK 失败：显示保留姿态，不计成功残差'}${solved.anomaly ? ' · 本帧异常' : ''}\n关节跳变 ${solved.joint_jump_deg.toFixed(2)}° · 躯干漂移 ${solved.torso_drift_deg.toFixed(2)}°\n限位 ${solved.limit_violations.length} 项 · 求解 ${(solved.solve_s * 1000).toFixed(1)} ms\n整条轨迹 ${manifest.summary.success_frames}/${manifest.summary.frames} 帧成功`;
  plottedGrippers(); save();
}
async function seek(i) {
  if (!trajectory) return;
  pause(); const serial = ++seekSerial, token = generation; seeking = true; $('#sync').textContent = '等待可用视频定位…';
  const targetIndex = Math.max(0, Math.min(trajectory.frames.length - 1, i)), frame = trajectory.frames[targetIndex];
  try {
    await videos.seek(frame.sample_timestamp);
    if (serial !== seekSerial || token !== generation) return;
    await display(targetIndex, video.currentTime);
  } catch (error) { showError(error); } finally { if (serial === seekSerial) seeking = false; }
}
async function loadCamera(preview = false) {
  if (!episode) return;
  const token = generation, time = trajectory?.frames[index]?.sample_timestamp || 0;
  pause();
  const current = await videos.load(episode.videos, $('#camera').value, time, preview);
  if (!current || token !== generation) return;
  if (trajectory) await display(index, video.currentTime);
  save(); updateMediaStatus();
}

async function loadEpisode(id, restore = false) {
  const restoreTime = restore ? stored.time || 0 : 0;
  pause(); generation++; controller.abort(); controller = new AbortController(); const token = generation;
  trajectory = manifest = jobKey = null; chunks = new Map(); index = 0; seeking = false; seekSerial++;
  videos.clear(); scene.setReferenceFrame(null);
  $('#error').textContent = ''; $('#loading').textContent = '读取轨迹…'; $('#diagnosis').textContent = '尚未计算。';
  for (const id of ['play', 'previous', 'next', 'seek', 'anomaly', 'compute']) $('#' + id).disabled = true;
  scene.clear(scene.targets); scene.clear(scene.trails); if (model) scene.pose(model.initial_snapshot, 'world');
  try {
    episode = await api('/api/episodes/' + id, { signal: controller.signal });
    if (token !== generation) return;
    $('#title').textContent = episode.task; $('#subtitle').textContent = `${episode.dataset} / ${episode.profile} / episode_${String(episode.episode).padStart(6, '0')}`;
    $('#notice').textContent = `${episode.basis_warning}。${labels[episode.review.status]}${episode.alignment_policy ? '' : '：' + episode.review.reason}`;
    $('#gripper-title').textContent = episode.dataset === 'HABIT' ? '夹爪 / 保存值（开闭方向未声明）' : '夹爪 / 0 开 · 100 闭';
    $('#provenance').textContent = JSON.stringify(episode, null, 2);
    scene.setReferenceFrame(episode);
    $('#compute').disabled = !metricAllowed();
    const selection = selectCameraSlots(episode.videos, restore ? stored.camera : null);
    $('#camera').innerHTML = selection.heads.map(v => `<option value="${esc(v.camera)}">${esc(v.camera.replace('observation.images.', ''))}</option>`).join('');
    $('#camera').value = selection.head?.camera || '';
    const cameraLoading = loadCamera().catch(showError);
    trajectory = await api(`/api/episodes/${id}/trajectory`, { signal: controller.signal });
    if (token !== generation) return;
    scene.setReferenceFrame(episode, trajectory.alignment);
    $('#provenance').textContent = JSON.stringify({episode, alignment: trajectory.alignment}, null, 2);
    await cameraLoading;
    if (token !== generation) return;
    scene.trajectory(trajectory.frames, selectedMode, metricAllowed());
    $('#seek').max = trajectory.frames.length - 1;
    for (const id of ['previous', 'next', 'seek']) $('#' + id).disabled = false;
    updateMediaStatus();
    $('#loading').textContent = metricAllowed() ? '目标已载入 · 等待计算' : '仅视频与原始数值 · ' + labels[episode.review.status];
    index = restore ? nearestFrame(trajectory.frames.map(f => f.sample_timestamp), restoreTime) : 0;
    await display(index, video.currentTime);
    if (restoreTime) await seek(index);
    document.querySelectorAll('#catalog button').forEach(b => b.classList.toggle('active', b.dataset.id === id));
    if (metricAllowed()) await compute();
  } catch (error) { showError(error); }
}

async function compute() {
  if (!episode || !metricAllowed() || !trajectory) return;
  const token = generation; $('#compute').disabled = true;
  try {
    let job = await api(`/api/episodes/${episode.id}/compute`, { method: 'POST', body: JSON.stringify({ mode: selectedMode }), signal: controller.signal });
    while (job.status === 'queued' || job.status === 'running') {
      $('#loading').textContent = job.status === 'queued' ? '等待共享计算队列…' : `计算 ${job.completed_frames || 0}/${job.total_frames || trajectory.frames.length} 帧`;
      $('#diagnosis').textContent = '计算中；视频播放和目标检查可独立进行。';
      await new Promise(resolve => setTimeout(resolve, 900)); if (token !== generation) return;
      job = await api('/api/jobs/' + job.key, { signal: controller.signal });
    }
    if (token !== generation) return;
    if (job.status !== 'complete') throw new Error(job.error || '计算失败，可重试以恢复检查点');
    jobKey = job.key; manifest = await api('/api/results/' + jobKey, { signal: controller.signal }); chunks = new Map();
    $('#loading').textContent = job.cache_hit ? '缓存已就绪' : '计算已完成'; $('#anomaly').disabled = !manifest.summary.anomaly_frames.length;
    $('#provenance').textContent = JSON.stringify({ episode, result: manifest }, null, 2);
    await display(index, presented); await loadCatalog();
  } catch (error) { showError(error); $('#loading').textContent = '计算未完成'; } finally { if (token === generation) $('#compute').disabled = !metricAllowed(); }
}

async function loadCatalog() {
  const token = ++catalogGeneration;
  const query = new URLSearchParams({ q: $('#search').value, source: $('#source').value, group: $('#group').value, review: $('#filter').value, processing: $('#processing').value, mode: selectedMode, offset, limit: 40, all: ['excluded', 'deferred'].includes($('#filter').value) ? '1' : '0' });
  const data = await api('/api/catalog?' + query);
  if (token !== catalogGeneration) return;
  total = data.total; $('#count').textContent = `${total.toLocaleString()} 条轨迹`; $('#page-number').textContent = `${Math.floor(offset / 40) + 1} / ${Math.max(1, Math.ceil(total / 40))}`;
  $('#prev-page').disabled = offset === 0; $('#next-page').disabled = offset + 40 >= total;
  const sourceValue = $('#source').value, groupValue = $('#group').value;
  $('#source').innerHTML = '<option value="">全部来源</option>' + (data.sources || []).map(s => `<option value="${esc(s.id)}">${esc(s.label)}</option>`).join('');
  $('#source').value = sourceValue;
  $('#source').title = data.sources?.find(s => s.id === sourceValue)?.label || '全部来源';
  const groups = (data.group_options || data.groups.map(id => ({id, label:id}))).filter(g => !sourceValue || g.source_id === sourceValue);
  $('#group').innerHTML = '<option value="">全部类别</option>' + groups.map(g => `<option value="${esc(g.id)}">${esc(g.label)}${g.episodes !== undefined ? ` · ${g.episodes} 条` : ''}</option>`).join('');
  $('#group').value = groups.some(g => g.id === groupValue) ? groupValue : '';
  const rows = data.episodes;
  $('#catalog').innerHTML = rows.map(row => `<button data-id="${row.id}" class="${episode?.id === row.id ? 'active' : ''}" ><strong>${esc(row.profile)} · ${row.episode}</strong><small>${esc(row.root.split('/').slice(-1)[0] || row.dataset)}</small><small>${esc(row.dataset)}</small><span class="pill ${row.review.status === 'admitted' ? '' : 'pending'}">${labels[row.review.status]}</span><small>${({complete:'缓存已完成',queued:'排队中',running:'计算中',failed:'计算失败',uncomputed:'尚未计算'}[row.computation?.status] || '')}${row.computation?.anomaly_count ? ` · ${row.computation.anomaly_count} 异常帧` : ''}</small></button>`).join('') || '<div class="empty">没有匹配的轨迹。调整筛选条件查看其他类别。</div>';
  document.querySelectorAll('#catalog button').forEach(button => button.addEventListener('click', () => loadEpisode(button.dataset.id)));
}

$('#play').onclick = async () => { if (videos.playIntent) await seek(index); else { try { await videos.play(); } catch (e) { showError(e); } } };
$('#previous').onclick = () => seek(index - 1); $('#next').onclick = () => seek(index + 1);
$('#anomaly').onclick = () => seek(nextAnomaly(manifest?.summary.anomaly_frames || [], index));
$('#seek').oninput = event => seek(+event.target.value);
$('#compute').onclick = compute;
$('#speed').onchange = () => { videos.setRate(+$('#speed').value); save(); };
videos.setRate(+$('#speed').value);
$('#camera').onchange = () => loadCamera().catch(showError); $('#preview').onclick = () => loadCamera(true).catch(showError);
$('#action-layer').onchange = () => display(index);
$('#mode').onchange = async () => { selectedMode = $('#mode').value; save(); if (episode) await loadEpisode(episode.id, true); };
let searchTimer; $('#search').oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { offset = 0; loadCatalog().catch(showError); }, 220); };
$('#source').onchange = () => { $('#group').value = ''; offset = 0; loadCatalog().catch(showError); };
for (const id of ['group', 'filter', 'processing']) $('#' + id).onchange = () => { offset = 0; loadCatalog().catch(showError); };
$('#prev-page').onclick = () => { offset = Math.max(0, offset - 40); loadCatalog().catch(showError); };
$('#next-page').onclick = () => { offset += 40; loadCatalog().catch(showError); };
window.addEventListener('resize', plottedGrippers);
let wasConnected = true;
async function health() {
  try {
    const value = await api('/api/health');
    $('#connection').textContent = value.cloud_connected ? `● 云端已连接 · 队列 ${value.queue}` : '● SSH / 媒体断线，等待自动重连';
    $('#connection').classList.toggle('error-state', !value.cloud_connected);
    if (!wasConnected && value.cloud_connected && Object.values(videos.slots).some(slot => slot.state === 'error' || slot.video.error)) loadCamera().catch(showError);
    wasConnected = value.cloud_connected;
    if (value.cloud_connected) videos.renewLeases();
  } catch { $('#connection').textContent = '共享服务暂时不可用'; wasConnected = false; }
}
setInterval(health, 5000); health();
async function init() {
  const [metadata] = await Promise.all([api('/api/model'), loadCatalog()]); model = metadata;
  await scene.loadRobot(model.glb_url, model, model.initial_snapshot); scene.pose(model.initial_snapshot, 'world');
  $('#loading').textContent = '模型已就绪';
  if (stored.episode) await loadEpisode(stored.episode, true);
  else {
    const first = await api('/api/catalog?review=admitted&limit=1');
    if (first.episodes[0]) await loadEpisode(first.episodes[0].id);
  }
}
init().catch(showError);
