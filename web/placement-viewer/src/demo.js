// Deterministic synthetic events, never presented as measured dataset results.
export function demoBundle(modelMetadata = null) {
  const names = modelMetadata?.body_names ?? [
    'astribot_torso_base', 'torso', 'shoulder_left', 'elbow_left',
    'astribot_arm_left_link_7', 'shoulder_right', 'elbow_right', 'astribot_arm_right_link_7', 'head',
  ];
  const xyz = [[0, 0, 0], [0, 0, 0.8], [0, 0.23, 1.2], [0.25, 0.38, 0.92],
    [0.48, 0.31, 0.78], [0, -0.23, 1.2], [0.25, -0.38, 0.92], [0.48, -0.31, 0.78], [0, 0, 1.5]];
  const reference = modelMetadata?.reference_snapshot ?? { body_poses_world: xyz.map((p) => [...p, 1, 0, 0, 0]) };
  let seed = 3107;
  const random = () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; };
  const events = [];
  for (let episode = 0; episode < 24; episode += 1) {
    for (const side of ['left', 'right']) {
      for (const event_type of ['grasp', 'release']) {
        const sign = side === 'left' ? 1 : -1;
        const position = [0.54 + (event_type === 'release' ? 0.18 : 0) + (random() - 0.5) * 0.22,
          sign * (0.22 + (random() - 0.5) * 0.2), 0.77 + (random() - 0.5) * 0.1];
        events.push({ event_id: `demo:${episode}:${side}:${event_type}`, episode_id: `示例 ${String(episode + 1).padStart(2, '0')}`,
          side, event_type, timestamp_s: event_type === 'grasp' ? 2.4 : 5.8,
          tcp_pose_world: [...position, 1, 0, 0, 0], tcp_pose_base: [...position, 1, 0, 0, 0],
          base_pose_world: [0, 0, 0, 1, 0, 0, 0], speed_m_s: random() * 0.035,
          acceleration_m_s2: random() * 0.3, low_motion: random() > 0.25 });
      }
    }
  }
  for (const event of events) {
    event.detection_sources = ['gripper'];
    event.related_event_ids = [];
    event.episode_elapsed_s = event.timestamp_s;
    event.initial_phase = false;
    event.initial_distance_m = 0.25;
    event.exclusion_reasons = [];
  }
  const extra = [];
  for (const event of events.filter((e) => e.event_type === 'grasp')) {
    const make = (suffix, changes) => ({ ...event, event_id: `${event.event_id}:${suffix}`,
      detection_sources: ['motion'], related_event_ids: [], event_type: 'dwell',
      speed_m_s: 0.002, acceleration_m_s2: 0.01, low_motion: true, ...changes });
    // Some transitions have independently detected sustained stop evidence.
    if (Number(event.event_id.split(':')[1]) % 2 === 0) {
      const stop = make('stop', {});
      stop.related_event_ids = [event.event_id];
      event.related_event_ids.push(stop.event_id);
      event.detection_sources.push('motion');
      extra.push(stop);
    }
    // Use reproducible per-episode samples so narrowing the episode range
    // changes the spatial distribution instead of retaining a fixed hotspot.
    const pressPosition = [.53 + random() * .24,
      (event.side === 'left' ? 1 : -1) * (.18 + random() * .18), .72 + random() * .14];
    extra.push(make('press', { timestamp_s: 4.2, episode_elapsed_s: 4.2,
      tcp_pose_world: [...pressPosition, 1, 0, 0, 0],
      tcp_pose_base: [...pressPosition, 1, 0, 0, 0] }));
    extra.push(make('startup', { event_type: 'grasp', detection_sources: ['gripper'],
      timestamp_s: .3, episode_elapsed_s: .3, initial_phase: true, initial_distance_m: .005,
      exclusion_reasons: ['initial_pose', 'initial_time'] }));
    extra.push(make('initial-stop', { timestamp_s: .5, episode_elapsed_s: .5,
      initial_phase: true, initial_distance_m: 0, exclusion_reasons: ['initial_pose', 'initial_time'] }));
  }
  events.push(...extra);
  events.sort((a, b) => a.episode_id.localeCompare(b.episode_id)
    || a.timestamp_s - b.timestamp_s || a.side.localeCompare(b.side));
  return { schema_version: 2, robot_model: modelMetadata ? '/data/demo/robot.glb' : null,
    metadata: { units: 'meters', up_axis: 'Z', quaternion_order: 'wxyz', demo: true,
      body_names: names, base_body: modelMetadata?.base_body ?? names[0],
      reference_pose: modelMetadata?.reference_pose,
      joint_names: modelMetadata?.joint_names,
      detector: { initial_ignore_s: 1, initial_radius_m: .05, dwell_duration_s: .2,
        speed_threshold_m_s: .02, acceleration_threshold_m_s2: .2, coincidence_tolerance_s: .15 },
      tcp_calibration: '示例点为合成数据，仅用于演示界面；机器人显示参考姿态。' },
    reference_snapshot: reference, events };
}
