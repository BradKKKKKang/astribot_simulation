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
export const hasCandidateAnnotations = (bundle) => bundle.schema_version >= 2 && bundle.metadata.source_schema_version !== 1;
export const objectName = (event) => typeof event.object === 'string' && event.object.trim() ? event.object.trim() : null;
export const UNKNOWN_OBJECT_COLOR = '#8995a3';
export const OPERATION_TYPES = { gripper: ['grasp', 'release'], press: ['press'] };

export function operationType(bundle) {
  const meta = bundle.metadata;
  if (bundle.schema_version === 5) {
    if (!Array.isArray(bundle.display?.event_types) || !bundle.display.event_types.length
        || !bundle.display.event_types.every((kind) => typeof kind === 'string' && kind.length)
        || new Set(bundle.display.event_types).size !== bundle.display.event_types.length
        || !Array.isArray(bundle.display.default_event_types) || !bundle.display.event_labels
        || !bundle.display.event_types.every((kind) => typeof bundle.display.event_labels[kind] === 'string')
        || bundle.display.default_event_types.some((kind) => !bundle.display.event_types.includes(kind))) {
      throw new Error('任务缺少事件显示声明');
    }
    return meta.type ?? 'gripper';
  }
  const declared = [meta.type, meta.task_config?.type, meta.detector?.type].filter((type) => type !== undefined);
  if (declared.some((type) => !Object.hasOwn(OPERATION_TYPES, type)) || new Set(declared).size > 1) {
    throw new Error('任务 type 缺失、无效或不一致');
  }
  // Historical bundles used motion_detector and stored extra gripper points.
  // Read their original default as one fixed type; never expose a mode switch.
  return declared[0] ?? (meta.detector?.motion_detector === 'press' ? 'press' : 'gripper');
}

// Build from the whole task, before filtering, so episode/hand/type filters
// cannot recolor an object. Known block names use their natural colors.
export function objectCatalog(events) {
  const names = [...new Set(events.map(objectName))].sort((a, b) => {
    if (a === null) return 1;
    if (b === null) return -1;
    return a < b ? -1 : a > b ? 1 : 0;
  });
  const blocks = new Map([['红色方块', '#ec6b65'], ['黄色方块', '#e9bd4c'], ['蓝色方块', '#67a9ef']]);
  const used = new Set([UNKNOWN_OBJECT_COLOR, ...blocks.values()]);
  return names.map((name) => {
    let color = name === null ? UNKNOWN_OBJECT_COLOR : blocks.get(name);
    if (!color) {
      let hash = 2166136261;
      for (const character of name) hash = Math.imul(hash ^ character.codePointAt(0), 16777619) >>> 0;
      let hue = hash % 360;
      color = `hsl(${hue}, 65%, 62%)`;
      while (used.has(color)) {
        hue = (hue + 137.508) % 360;
        color = `hsl(${hue.toFixed(3)}, 65%, 62%)`;
      }
    }
    used.add(color);
    return { name, label: name ?? '未识别', color };
  });
}

export function exclusionReasons(event, filters) {
  if (filters.analysisSelection) {
    return event.analysis_selected === false
      ? (event.analysis_exclusion_reasons?.length ? event.analysis_exclusion_reasons : ['分析未选中']) : [];
  }
  const reasons = [];
  if (filters.excludeInitial !== false && event.initial_phase === true) reasons.push('初始位姿测试区');
  if (Number.isFinite(event.episode_elapsed_s) && event.episode_elapsed_s < (filters.ignoreSeconds ?? 0)) reasons.push('开头忽略时段');
  return reasons;
}

export function filterEvents(events, filters) {
  const type = filters.type ?? 'gripper';
  if (!filters.availableEventTypes && !Object.hasOwn(OPERATION_TYPES, type)) throw new Error('不支持的任务 type');
  return events.filter((event) => {
    const z = eventPosition(event, filters.frame)[2];
    return (!filters.episodeIds || filters.episodeIds.includes(event.episode_id))
      && (filters.side === 'all' || event.side === filters.side)
      && (!filters.objects || filters.objects.includes(objectName(event)))
      && (filters.availableEventTypes ?? OPERATION_TYPES[type]).includes(event.event_type)
      && (!filters.eventTypes || filters.eventTypes.includes(event.event_type))
      && (filters.includeExcluded || exclusionReasons(event, filters).length === 0)
      && z >= filters.zMin && z <= filters.zMax;
  });
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
  if (![1, 2, 3, 4, 5].includes(bundle?.schema_version) || !Array.isArray(bundle.events)) throw new Error('不支持的数据格式，请导入 placement.json');
  const meta = bundle.metadata;
  if (meta?.units !== 'meters' || meta.up_axis !== 'Z' || meta.quaternion_order !== 'wxyz') {
    throw new Error('数据必须声明米制、Z 轴朝上、wxyz 四元数，避免尺度与方向不一致');
  }
  operationType(bundle);
  if (!Array.isArray(meta.body_names) || !meta.body_names.length
      || new Set(meta.body_names).size !== meta.body_names.length
      || !meta.body_names.every((n) => typeof n === 'string')
      || !meta.body_names.includes(meta.base_body)) throw new Error('机器人 body_names / base_body 不完整');
  const checkSnapshot = (snapshot) => validateSnapshot(snapshot, meta);
  checkSnapshot(bundle.reference_snapshot);
  const ids = new Set();
  for (const event of bundle.events) {
    if (typeof event.event_id !== 'string' || ids.has(event.event_id)) throw new Error('event_id 缺失或重复');
    ids.add(event.event_id);
    if (typeof event.episode_id !== 'string' || !['left', 'right'].includes(event.side)
        || !(bundle.schema_version === 5 ? bundle.display.event_types : ['grasp', 'release', 'dwell', ...(bundle.schema_version >= 3 ? ['press'] : [])]).includes(event.event_type) || !Number.isFinite(event.timestamp_s)) {
      throw new Error('事件的 episode、左右手、类型或时间格式不正确');
    }
    validatePose(event.tcp_pose_world, 'TCP world');
    if (bundle.schema_version >= 4) {
      if (event.object !== null && typeof event.object !== 'string') throw new Error('物体名称格式不正确');
      const identification = event.identification;
      if (!['not_requested', 'skipped', 'success', 'failed', ...(bundle.schema_version === 5 ? ['unknown'] : [])].includes(identification?.status)) throw new Error('识别状态格式不正确');
    }
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
    if (hasCandidateAnnotations(bundle)) {
      if (!Array.isArray(event.detection_sources) || !event.detection_sources.length
          || !event.detection_sources.every((s) => ['gripper', 'motion'].includes(s))
          || (bundle.schema_version < 5 && !event.detection_sources.includes(isMotionEvent(event) ? 'motion' : 'gripper'))
          || !Array.isArray(event.related_event_ids) || !event.related_event_ids.every((id) => typeof id === 'string')
          || typeof event.initial_phase !== 'boolean'
          || !Number.isFinite(event.episode_elapsed_s) || event.episode_elapsed_s < 0
          || !Number.isFinite(event.initial_distance_m) || event.initial_distance_m < 0
          || !Array.isArray(event.exclusion_reasons)) throw new Error('候选点来源或初始阶段信息不完整');
    }
    if (event.event_type === 'press' && !['press_depth_m', 'press_retract_m', 'press_lateral_m']
      .every((field) => Number.isFinite(event[field]) && event[field] >= 0)) throw new Error('按压候选的位移依据不完整');
    if (event.snapshot) checkSnapshot(event.snapshot);
  }
  return bundle;
}

export function validateSnapshot(snapshot, metadata) {
  if (!Array.isArray(snapshot?.body_poses_world) || snapshot.body_poses_world.length !== metadata.body_names.length) {
    throw new Error('机器人姿态快照与 body_names 长度不一致');
  }
  snapshot.body_poses_world.forEach((pose) => validatePose(pose, 'body pose'));
  if (snapshot.joint_positions_rad !== undefined) {
    const count = metadata.joint_names?.length ?? snapshot.joint_positions_rad?.length;
    finiteArray(snapshot.joint_positions_rad, count, 'joint positions');
  }
  return snapshot;
}
