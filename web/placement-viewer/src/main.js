import './style.css';
import { EVENT_LABELS, eventPosition, exclusionReasons, filterEvents, validateBundle } from './data.js';
import { PlacementScene } from './scene.js';

const $ = (id) => document.getElementById(id);
const state = { bundle: null, events: [], selected: null, side: 'all', frame: 'base',
  episodes: [], episodeStart: 1, episodeEnd: 1,
  pointTypes: { gripper: new Set(['grasp', 'release']), union: new Set(['grasp', 'release', 'dwell', 'press']) },
  modelUrl: null, modelObjectUrl: null, pendingBundle: null, limit: 60, revision: 0 };
const tasks = { items: [], selected: '', revision: null, externalName: '', loading: false,
  polling: false, followLatest: !new URLSearchParams(window.location.search).has('dataset') };
let scene;
const sceneMessage = $('scene-message');

function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
  $('notice').hidden = !message;
}

function renderTasks() {
  const choices = tasks.items.map((task) => new Option(task.name, task.id));
  if (tasks.selected === 'external') choices.push(new Option(tasks.externalName, 'external'));
  if (!choices.length) choices.push(new Option('暂无已发布任务', ''));
  $('task-select').replaceChildren(...choices);
  $('task-select').value = tasks.selected;
}

async function loadSource(action) {
  if (tasks.loading) return;
  tasks.loading = true;
  ['task-select', 'bundle-file', 'model-file'].forEach((id) => { $(id).disabled = true; });
  try { await action(); }
  finally {
    tasks.loading = false;
    ['task-select', 'bundle-file', 'model-file'].forEach((id) => { $(id).disabled = false; });
    renderTasks();
    if (state.bundle) sceneMessage.hidden = true;
  }
}

async function loadBundleUrl(path, name, task = null) {
  const url = new URL(path, window.location.href);
  if (url.origin !== window.location.origin) throw new Error('数据 URL 必须与页面同源');
  const response = await fetch(url, { cache: 'no-store', headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`数据加载失败：HTTP ${response.status}`);
  const bundle = validateBundle(await response.json());
  const model = new URL(bundle.robot_model || 'robot.glb', url);
  if (model.origin !== window.location.origin) throw new Error('模型 URL 必须与页面同源');
  if (task) model.searchParams.set('v', task.revision);
  await openBundle(bundle, model.href, name, task?.id ?? 'external');
  tasks.revision = task?.revision ?? null;
}

async function refreshTasks() {
  if (tasks.polling || tasks.loading) return;
  tasks.polling = true;
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}api/tasks`,
      { cache: 'no-store', headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error(`任务列表加载失败：HTTP ${response.status}`);
    const catalog = await response.json();
    if (!Array.isArray(catalog.tasks) || catalog.tasks.some((task) =>
      !task.id || !task.name || !task.bundle || !task.revision)) throw new Error('任务列表格式无效');
    if (JSON.stringify(tasks.items) !== JSON.stringify(catalog.tasks)) {
      tasks.items = catalog.tasks;
      renderTasks();
    }
    const selected = tasks.items.find((task) => task.id === tasks.selected);
    const target = tasks.followLatest || (tasks.selected !== 'external' && !selected) ? tasks.items[0] : selected;
    if (!target && tasks.selected !== 'external' && !tasks.items.length) {
      tasks.selected = '';
      tasks.revision = null;
      state.bundle = null;
      state.events = [];
      state.selected = null;
      state.episodes = [];
      state.episodeStart = state.episodeEnd = 0;
      scene.clear(scene.robot);
      scene.renderEvents([], state.frame, 'points');
      scene.selected.visible = false;
      $('count-total').textContent = $('count-episodes').textContent = $('list-count').textContent = '0';
      syncEpisodeControls();
      renderList();
      details(null);
      renderTasks();
      sceneMessage.hidden = false;
      sceneMessage.textContent = '暂无已发布任务，请完成点位提取或导入数据。';
    }
    if (target && (target.id !== tasks.selected || target.revision !== tasks.revision)) {
      await loadSource(() => loadBundleUrl(`${import.meta.env.BASE_URL}${target.bundle}`, target.name, target));
    }
  } finally { tasks.polling = false; }
}

function filters() {
  const mode = $('candidate-mode').value;
  return { episodeIds: state.episodes.slice(state.episodeStart - 1, state.episodeEnd),
    episodeStart: state.episodeStart, episodeEnd: state.episodeEnd,
    eventTypes: mode === 'motion' ? ['dwell', 'press'] : [...state.pointTypes[mode]],
    side: state.side, frame: $('coordinate-frame').value,
    candidateMode: $('candidate-mode').value, excludeInitial: $('exclude-initial').checked,
    ignoreSeconds: Number($('ignore-seconds').value), includeExcluded: $('include-excluded').checked,
    zMin: Number($('z-min').value), zMax: Number($('z-max').value) };
}

function syncEpisodeControls() {
  const total = state.episodes.length;
  for (const bound of ['start', 'end']) {
    const value = bound === 'start' ? state.episodeStart : state.episodeEnd;
    for (const suffix of ['', '-range']) {
      const control = $(`episode-${bound}${suffix}`);
      control.min = total ? '1' : '0';
      control.max = String(total || 1);
      control.value = String(value);
      control.disabled = total < 2 && (suffix === '-range' || !total);
    }
  }
  const denominator = Math.max(1, total - 1);
  const left = total ? (state.episodeStart - 1) / denominator * 100 : 0;
  const right = total > 1 ? (state.episodeEnd - 1) / denominator * 100 : left;
  $('episode-range-fill').style.left = `${left}%`;
  $('episode-range-fill').style.width = `${right - left}%`;
  // The lower thumb must remain reachable when both handles overlap at the end.
  $('episode-start-range').style.zIndex = state.episodeStart === total ? '3' : '2';
  $('episode-range-label').textContent = total ? `${state.episodeStart}–${state.episodeEnd} / ${total}` : '无 episode';
  $('episode-names').textContent = total ? `${state.episodes[state.episodeStart - 1]} → ${state.episodes[state.episodeEnd - 1]}` : '';
}

function updateEpisodeRange(bound, raw) {
  const value = Number(raw);
  if (!raw || !Number.isInteger(value) || !state.episodes.length) {
    syncEpisodeControls();
    return;
  }
  if (bound === 'start') state.episodeStart = Math.max(1, Math.min(value, state.episodeEnd));
  else state.episodeEnd = Math.max(state.episodeStart, Math.min(value, state.episodes.length));
  syncEpisodeControls();
  refresh();
}

function renderPointChoices(mode) {
  const legend = $('candidate-legend');
  const motionType = state.bundle?.metadata.detector?.motion_detector === 'press' ? 'press' : 'dwell';
  if (legend.dataset.mode === `${mode}:${motionType}`) return;
  legend.dataset.mode = `${mode}:${motionType}`;
  const types = mode === 'gripper' ? ['grasp', 'release'] : mode === 'motion' ? [motionType] : ['grasp', 'release', motionType];
  legend.replaceChildren(...types.map((type) => {
    const item = document.createElement(mode === 'motion' ? 'span' : 'label');
    if (mode !== 'motion') {
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.id = `show-${type}`;
      input.checked = state.pointTypes[mode].has(type);
      input.addEventListener('change', () => {
        if (input.checked) state.pointTypes[mode].add(type);
        else state.pointTypes[mode].delete(type);
        refresh();
      });
      item.append(input);
    }
    const dot = document.createElement('i');
    dot.className = `dot ${type}`;
    item.append(dot, `${EVENT_LABELS[type]}点`);
    return item;
  }));
}

function details(event) {
  const demo = state.bundle?.metadata.demo;
  $('detail-title').textContent = event
    ? `${EVENT_LABELS[event.event_type]} · ${event.side === 'left' ? '左手' : '右手'}`
    : '选择一个操作候选点';
  const reasons = event ? exclusionReasons(event, filters()) : [];
  const motionEvidence = event?.event_type === 'press'
    ? `下压 ${(event.press_depth_m * 1000).toFixed(1)} mm → 低点 → 抬起 ${(event.press_retract_m * 1000).toFixed(1)} mm，横向偏移 ${(event.press_lateral_m * 1000).toFixed(1)} mm`
    : '持续低速且低加速度';
  $('detail-evidence').textContent = event ? `依据：${(event.detection_sources ?? ['gripper']).map((s) => s === 'gripper' ? '夹爪开合' : motionEvidence).join(' + ')}${Number.isFinite(event.transition_start_s) ? ` · 区间 ${event.transition_start_s.toFixed(3)}–${event.transition_end_s.toFixed(3)} s` : ''}${reasons.length ? ` · 已排除：${reasons.join('、')}` : ''}` : '';
  $('detail-summary').textContent = event
    ? `${event.episode_id} · ${event.snapshot ? '已显示事件发生时的机器人姿态' : demo ? '合成示例点 · 机器人为参考姿态' : '无事件姿态快照，显示参考姿态'}`
    : '点位与机器人共享相同坐标和米制尺度。';
  const values = event ? [
    `${eventPosition(event, state.frame).map((p) => p.toFixed(3)).join(' / ')} m`,
    `${event.timestamp_s.toFixed(3)} s`,
    event.speed_m_s == null ? '—' : `${event.speed_m_s.toFixed(4)} m/s`,
    event.acceleration_m_s2 == null ? '—' : `${event.acceleration_m_s2.toFixed(4)} m/s²`,
  ] : ['—', '—', '—', '—'];
  $('event-detail').querySelectorAll('dd').forEach((node, i) => { node.textContent = values[i]; });
}

function select(event) {
  state.selected = event;
  scene.select(event, state.frame);
  scene.pose(event?.snapshot ?? state.bundle.reference_snapshot, state.frame);
  details(event);
  $('event-list').querySelectorAll('button').forEach((button) => {
    button.classList.toggle('selected', button.dataset.eventId === event?.event_id);
  });
}

function renderList() {
  const container = $('event-list');
  container.replaceChildren();
  const visible = state.events.slice(0, state.limit);
  for (const event of visible) {
    const button = document.createElement('button');
    button.className = 'event-row';
    button.dataset.eventId = event.event_id;
    button.classList.toggle('selected', state.selected?.event_id === event.event_id);
    const kind = document.createElement('span');
    kind.className = 'event-kind';
    const dot = document.createElement('span');
    dot.className = `dot ${event.event_type}`;
    kind.append(dot, EVENT_LABELS[event.event_type]);
    const reasons = exclusionReasons(event, filters());
    if (reasons.length) {
      button.classList.add('excluded');
      button.title = `排除原因：${reasons.join('、')}`;
      const badge = document.createElement('small');
      badge.textContent = '排除';
      kind.append(badge);
    }
    const episode = document.createElement('span');
    episode.className = 'episode-name';
    episode.textContent = event.episode_id;
    const side = document.createElement('span');
    side.textContent = event.side === 'left' ? '左手' : '右手';
    const time = document.createElement('time');
    time.textContent = `${event.timestamp_s.toFixed(2)} s`;
    const coordinates = document.createElement('span');
    coordinates.className = 'coords';
    coordinates.textContent = eventPosition(event, state.frame).map((p) => p.toFixed(2)).join(' / ');
    button.append(kind, episode, side, time, coordinates);
    button.addEventListener('click', () => select(event));
    container.append(button);
  }
  if (!state.events.length) {
    const message = document.createElement('p');
    message.className = 'empty-list';
    message.textContent = '当前条件下没有操作候选点，请调整筛选范围。';
    container.append(message);
  }
  $('more-events').hidden = state.events.length <= state.limit;
}

function refresh() {
  if (!state.bundle) return;
  const active = filters();
  if (!Number.isFinite(active.ignoreSeconds) || active.ignoreSeconds < 0) {
    notice('忽略开头时长必须为非负数。', true);
    return;
  }
  if (!Number.isFinite(active.zMin) || !Number.isFinite(active.zMax) || active.zMin > active.zMax) {
    notice('高度范围无效：最低高度不能大于最高高度。', true);
    return;
  }
  const frameChanged = state.frame !== active.frame;
  state.frame = active.frame;
  scene.frame = active.frame;
  state.events = filterEvents(state.bundle.events, active);
  const counted = filterEvents(state.bundle.events, { ...active, includeExcluded: false });
  const excludedIds = new Set(state.events.filter((e) => exclusionReasons(e, active).length).map((e) => e.event_id));
  $('exclusion-summary').textContent = state.bundle.schema_version === 1
    ? '旧版文件缺少停留与初始阶段信息，请重新提取；不会把低速帧假定为停留。'
    : `每只手离开初始位置 ${((state.bundle.metadata.detector?.initial_radius_m ?? .05) * 100).toFixed(0)} cm 并持续 ${state.bundle.metadata.detector?.initial_departure_duration_s ?? .15} s 后解除。${state.bundle.events.filter((e) => exclusionReasons(e, active).length).length} 个原始候选命中排除规则；灰色检查点不计入热力图。`;
  const help = { gripper: '使用夹爪开合变化定位抓取与释放。', motion: '每次持续停留取一个点，不判断为抓取或真实接触。',
    union: '合并两种候选；同侧关联的运动和开合点只计一次。' };
  const detector = state.bundle.metadata.detector ?? {};
  const press = detector.motion_detector === 'press';
  if (press) help.motion = '每次下压到低点、随后抬起取一个候选点；不代表已确认接触。';
  $('candidate-help').textContent = help[active.candidateMode] + (active.candidateMode === 'gripper' ? ''
    : press ? ` 下压 ≥ ${detector.press_min_depth_m * 1000} mm，抬起 ≥ ${detector.press_min_retract_m * 1000} mm，横向偏移 ≤ ${detector.press_max_lateral_m * 1000} mm。`
      : ` 停留阈值：v ≤ ${detector.speed_threshold_m_s ?? '—'} m/s，a ≤ ${detector.acceleration_threshold_m_s2 ?? '—'} m/s²，持续 ${detector.dwell_duration_s ?? '—'} s。`);
  renderPointChoices(active.candidateMode);
  const mode = $('display-mode').value;
  const isField = mode.startsWith('field');
  $('field-control').hidden = $('field-help').hidden = !isField;
  document.querySelector('.density-legend').hidden = !isField;
  scene.renderEvents(state.events, active.frame, mode, counted, excludedIds, Number($('bandwidth').value) / 100);
  $('density-max').textContent = counted.length ? '100%' : '0%';
  $('bandwidth-label').textContent = `${$('bandwidth').value} cm`;
  $('opacity-label').textContent = `${$('robot-opacity').value}%`;
  scene.opacity(Number($('robot-opacity').value) / 100);
  scene.grid.visible = $('show-grid').checked;
  $('count-total').textContent = counted.length.toLocaleString();
  $('count-episodes').textContent = new Set(counted.map((event) => event.episode_id)).size.toLocaleString();
  $('list-count').textContent = String(state.events.length);
  $('frame-badge').textContent = state.frame === 'base' ? 'BASE FRAME' : 'WORLD FRAME';
  if (!state.events.includes(state.selected)) state.selected = null;
  select(state.selected);
  renderList();
  if (frameChanged) scene.fit();
}

async function openBundle(bundle, modelUrl, sourceName, sourceId = 'external') {
  validateBundle(bundle);
  const revision = ++state.revision;
  sceneMessage.hidden = false;
  sceneMessage.textContent = '正在加载机器人模型…';
  await scene.loadRobot(modelUrl, bundle.metadata, bundle.reference_snapshot);
  if (revision !== state.revision) return;
  state.bundle = bundle;
  state.modelUrl = modelUrl;
  state.selected = null;
  state.limit = 60;
  $('ignore-seconds').value = String(bundle.metadata.detector?.initial_ignore_s ?? 1);
  $('candidate-mode').querySelectorAll('option').forEach((option) => {
    option.disabled = bundle.schema_version === 1 && option.value !== 'gripper';
  });
  if (bundle.schema_version === 1) $('candidate-mode').value = 'gripper';
  else if (bundle.metadata.detector?.motion_detector === 'press') $('candidate-mode').value = 'motion';
  $('candidate-mode').querySelector('option[value="motion"]').textContent = bundle.metadata.detector?.motion_detector === 'press'
    ? '下压后抬起 · 按压候选' : '持续低速且低加速度 · 停留候选';
  ['exclude-initial', 'ignore-seconds', 'include-excluded'].forEach((id) => { $(id).disabled = bundle.schema_version === 1; });
  state.episodes = [...new Set([...(bundle.metadata.episodes ?? []).map((episode) => episode.episode_id),
    ...bundle.events.map((event) => event.episode_id)])].sort((a, b) => a.localeCompare(b, 'zh-CN', { numeric: true }));
  state.episodeStart = state.episodes.length ? 1 : 0;
  state.episodeEnd = state.episodes.length;
  syncEpisodeControls();
  tasks.selected = sourceId;
  tasks.externalName = sourceName;
  renderTasks();
  sceneMessage.hidden = true;
  notice(bundle.metadata.demo ? '当前为合成示例点，仅用于演示空间分布；不代表真实轨迹统计。'
    : bundle.metadata.tcp_calibration || '已加载事件数据。');
  refresh();
  scene.fit();
}

async function importFile(file) {
  if (!file) return;
  try {
    const bundle = validateBundle(JSON.parse(await file.text()));
    state.pendingBundle = { bundle, filename: file.name };
    notice('事件文件已校验，请选择同一次导出的 robot.glb，以保证机器人模型与点位对齐。');
    $('model-file').value = '';
    $('model-file').click();
  } catch (error) { notice(`导入失败：${error.message}`, true); }
}

async function importModel(file) {
  if (!file || !state.pendingBundle) return;
  const objectUrl = URL.createObjectURL(file);
  const oldUrl = state.modelObjectUrl;
  try {
    const { bundle, filename } = state.pendingBundle;
    tasks.followLatest = false;
    await loadSource(() => openBundle(bundle, objectUrl, filename));
    state.modelObjectUrl = objectUrl;
    state.pendingBundle = null;
    if (oldUrl) URL.revokeObjectURL(oldUrl);
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    sceneMessage.hidden = true;
    notice(`模型导入失败：${error.message}。请重新导入事件文件和对应模型。`, true);
  }
}

async function boot() {
try {
  scene = new PlacementScene($('viewport'), (event) => select(event));
  $('side-filter').querySelectorAll('button').forEach((button) => button.addEventListener('click', () => {
    state.side = button.dataset.value;
    $('side-filter').querySelectorAll('button').forEach((b) => b.classList.toggle('active', b === button));
    refresh();
  }));
  ['coordinate-frame', 'display-mode', 'bandwidth',
    'z-min', 'z-max', 'robot-opacity', 'show-grid', 'candidate-mode', 'exclude-initial',
    'ignore-seconds', 'include-excluded'].forEach((id) => $(id).addEventListener('input', refresh));
  for (const bound of ['start', 'end']) {
    $(`episode-${bound}-range`).addEventListener('input', (event) => updateEpisodeRange(bound, event.target.value));
    $(`episode-${bound}`).addEventListener('change', (event) => updateEpisodeRange(bound, event.target.value));
  }
  document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => {
    scene.fit(button.dataset.view);
    document.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('active', b === button));
  }));
  $('fit-view').addEventListener('click', () => scene.fit());
  $('bundle-file').addEventListener('change', (event) => importFile(event.target.files[0]));
  $('model-file').addEventListener('change', (event) => importModel(event.target.files[0]));
  $('task-select').addEventListener('change', (event) => {
    tasks.followLatest = false;
    const task = tasks.items.find((item) => item.id === event.target.value);
    if (task) loadSource(() => loadBundleUrl(`${import.meta.env.BASE_URL}${task.bundle}`, task.name, task))
      .catch((error) => notice(`任务加载失败：${error.message}`, true));
  });
  $('more-events').addEventListener('click', () => { state.limit += 100; renderList(); });
  $('export-events').addEventListener('click', () => {
    if (!state.bundle) return;
    const output = { ...state.bundle, events: state.events,
      metadata: { ...state.bundle.metadata, view_filters: filters() } };
    const url = URL.createObjectURL(new Blob([JSON.stringify(output)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'placement-filtered.json';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  // A URL query can select a same-origin static bundle; no control endpoint exists.
  const dataset = new URLSearchParams(window.location.search).get('dataset');
  if (dataset) {
    await loadSource(() => loadBundleUrl(dataset, new URL(dataset, window.location.href).pathname.split('/').at(-1)));
  }
  const pollTasks = () => refreshTasks().catch((error) => notice(`自动更新失败：${error.message}`, true));
  await pollTasks();
  if (!state.bundle) sceneMessage.textContent = '暂无已发布任务，请完成点位提取或导入数据。';
  setInterval(pollTasks, 5000);
} catch (error) {
  sceneMessage.hidden = false;
  sceneMessage.textContent = `三维场景加载失败：${error.message}`;
  notice('请确认浏览器支持 WebGL；数据和 robot.glb 可通过右上角重新导入。', true);
  console.error(error);
}
}

boot();
