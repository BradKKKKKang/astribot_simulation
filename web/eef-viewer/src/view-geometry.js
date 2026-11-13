import * as THREE from 'three';

export function alignRobotToChassis(robot, chassisToUsd) {
  // FK bodies are in USD coordinates; targets and achieved EEF are already in
  // SDK chassis coordinates. Only the model needs the inverse translation.
  robot.position.fromArray(chassisToUsd).negate();
  robot.updateWorldMatrix(true, true);
  const bounds = new THREE.Box3().setFromObject(robot, true);
  if (bounds.isEmpty() || !Number.isFinite(bounds.min.z)) throw new Error('无法确定机器人几何最低点');
  return { groundZ: bounds.min.z };
}

export function referenceDescription(episode, alignment) {
  const reference = episode?.reference_frames;
  return {
    name: episode?.coordinates?.target_frame || episode?.coordinates?.reference_frame || '未声明',
    chassisOffset: reference?.chassis_to_usd_translation_m || [0, 0, -.097],
    dataToChassis: reference?.data_to_chassis || [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    showData: episode?.review?.status === 'admitted',
    alignment: alignment || {mode: reference?.alignment_status || 'chassis', per_arm:{}},
    explicit: Boolean(episode?.alignment_policy),
    assumption: '数据参考系暂与 SDK chassis 同原点、同轴向；未做跨本体标定',
  };
}
