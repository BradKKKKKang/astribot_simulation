import { Matrix4, Quaternion, Vector3 } from 'three';

export function poseMatrix(pose) {
  const [x, y, z, w, qx, qy, qz] = pose;
  return new Matrix4().compose(new Vector3(x, y, z), new Quaternion(qx, qy, qz, w), new Vector3(1, 1, 1));
}

export function eventPosition(event, frame) {
  return (frame === 'base' ? event.tcp_pose_base : event.tcp_pose_world).slice(0, 3);
}

export function snapshotPoses(snapshot, metadata, frame) {
  const poses = snapshot.body_poses_world;
  const baseIndex = metadata.body_names.indexOf(metadata.base_body);
  const inverse = frame === 'base' ? poseMatrix(poses[baseIndex]).invert() : new Matrix4();
  return poses.map((pose) => inverse.clone().multiply(poseMatrix(pose)));
}

export const EVENT_LABELS = { grasp: '抓取', release: '释放', dwell: '停留', press: '按压候选' };
export const isMotionEvent = (event) => ['dwell', 'press'].includes(event.event_type);

export function exclusionReasons(event, filters) {
  const reasons = [];
  if (filters.excludeInitial !== false && event.initial_phase === true) reasons.push('初始位姿测试区');
  if (Number.isFinite(event.episode_elapsed_s) && event.episode_elapsed_s < (filters.ignoreSeconds ?? 0)) reasons.push('开头忽略时段');
  return reasons;
}

export function filterEvents(events, filters) {
  const mode = filters.candidateMode ?? 'gripper';
  if (!['gripper', 'motion', 'union'].includes(mode)) throw new Error('不支持的候选点依据');
  const selected = events.filter((event) => {
    const z = eventPosition(event, filters.frame)[2];
    const gripper = !isMotionEvent(event);
    const matchesMode = mode === 'gripper' ? gripper
      : mode === 'motion' ? isMotionEvent(event) : true;
    return (!filters.episodeIds || filters.episodeIds.includes(event.episode_id))
      && (filters.side === 'all' || event.side === filters.side)
      && matchesMode
      && (mode === 'motion' || !filters.eventTypes || filters.eventTypes.includes(event.event_type))
      && (filters.includeExcluded || exclusionReasons(event, filters).length === 0)
      && z >= filters.zMin && z <= filters.zMax;
  });
  // In union mode, co-detected stop/transition candidates count once. Never
  // merge different hands or episodes, and preserve independent motion stops.
  if (mode !== 'union') return selected;
  const transitions = new Map(selected.filter((e) => !isMotionEvent(e)).map((e) => [e.event_id, e]));
  return selected.filter((event) => !isMotionEvent(event) || !(event.related_event_ids ?? []).some((id) => {
    const linked = transitions.get(id);
    return linked?.side === event.side && linked?.episode_id === event.episode_id
      && (exclusionReasons(linked, filters).length === 0 || exclusionReasons(event, filters).length > 0);
  }));
}

function finiteArray(value, length, name) {
  if (!Array.isArray(value) || value.length !== length || !value.every(Number.isFinite)) {
    throw new Error(`${name} 格式不正确：需要 ${length} 个有限数值`);
  }
}

function validatePose(pose, name) {
  finiteArray(pose, 7, name);
  if (Math.abs(Math.hypot(...pose.slice(3)) - 1) > 0.005) throw new Error(`${name} 四元数未归一化`);
}

export function validateBundle(bundle) {
  if (![1, 2, 3].includes(bundle?.schema_version) || !Array.isArray(bundle.events)) throw new Error('不支持的数据格式，请导入 placement.json');
  const meta = bundle.metadata;
  if (meta?.units !== 'meters' || meta.up_axis !== 'Z' || meta.quaternion_order !== 'wxyz') {
    throw new Error('数据必须声明米制、Z 轴朝上、wxyz 四元数，避免尺度与方向不一致');
  }
  if (!Array.isArray(meta.body_names) || !meta.body_names.length
      || new Set(meta.body_names).size !== meta.body_names.length
      || !meta.body_names.every((n) => typeof n === 'string')
      || !meta.body_names.includes(meta.base_body)) throw new Error('机器人 body_names / base_body 不完整');
  const validateSnapshot = (snapshot) => {
    if (!Array.isArray(snapshot?.body_poses_world) || snapshot.body_poses_world.length !== meta.body_names.length) {
      throw new Error('机器人姿态快照与 body_names 长度不一致');
    }
    snapshot.body_poses_world.forEach((p) => validatePose(p, 'body pose'));
  };
  validateSnapshot(bundle.reference_snapshot);
  const ids = new Set();
  for (const event of bundle.events) {
    if (typeof event.event_id !== 'string' || ids.has(event.event_id)) throw new Error('event_id 缺失或重复');
    ids.add(event.event_id);
    if (typeof event.episode_id !== 'string' || !['left', 'right'].includes(event.side)
        || !['grasp', 'release', 'dwell', ...(bundle.schema_version === 3 ? ['press'] : [])].includes(event.event_type) || !Number.isFinite(event.timestamp_s)) {
      throw new Error('事件的 episode、左右手、类型或时间格式不正确');
    }
    validatePose(event.tcp_pose_world, 'TCP world');
    validatePose(event.tcp_pose_base, 'TCP base');
    validatePose(event.base_pose_world, 'base world');
    // Catch inconsistent reference frames before drawing convincing-looking but
    // incorrectly aligned heatmaps.
    const expected = new Vector3(...event.tcp_pose_world.slice(0, 3))
      .applyMatrix4(poseMatrix(event.base_pose_world).invert());
    if (expected.distanceTo(new Vector3(...event.tcp_pose_base.slice(0, 3))) > 0.0001) {
      throw new Error('事件的世界坐标与底座坐标不一致');
    }
    for (const field of ['speed_m_s', 'acceleration_m_s2']) {
      if (event[field] != null && (!Number.isFinite(event[field]) || event[field] < 0)) throw new Error(`无效的 ${field}`);
    }
    if (bundle.schema_version >= 2) {
      if (!Array.isArray(event.detection_sources) || !event.detection_sources.length
          || !event.detection_sources.every((s) => ['gripper', 'motion'].includes(s))
          || !event.detection_sources.includes(isMotionEvent(event) ? 'motion' : 'gripper')
          || !Array.isArray(event.related_event_ids) || !event.related_event_ids.every((id) => typeof id === 'string')
          || typeof event.initial_phase !== 'boolean'
          || !Number.isFinite(event.episode_elapsed_s) || event.episode_elapsed_s < 0
          || !Number.isFinite(event.initial_distance_m) || event.initial_distance_m < 0
          || !Array.isArray(event.exclusion_reasons)) throw new Error('候选点来源或初始阶段信息不完整');
    }
    if (event.event_type === 'press' && !['press_depth_m', 'press_retract_m', 'press_lateral_m']
      .every((field) => Number.isFinite(event[field]) && event[field] >= 0)) throw new Error('按压候选的位移依据不完整');
    if (event.snapshot) validateSnapshot(event.snapshot);
  }
  return bundle;
}
