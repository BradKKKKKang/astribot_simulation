import test from 'node:test';
import assert from 'node:assert/strict';
import { Vector3 } from 'three';
import { eventPosition, exclusionReasons, filterEvents, operationType, poseMatrix, snapshotPoses, validateBundle } from '../src/data.js';
import { demoBundle } from '../src/demo.js';

test('demo retains all candidate sources and filters hands independently', () => {
  const bundle = validateBundle(demoBundle());
  assert.equal(bundle.events.length, 264);
  const filters = { episode: 'all', side: 'left', frame: 'base',
    zMin: -2, zMax: 3 };
  const events = filterEvents(bundle.events, filters);
  assert.equal(events.length, 48);
  assert(events.every((e) => e.side === 'left' && e.event_type !== 'dwell'));
  assert.equal(filterEvents(bundle.events, { ...filters, zMax: 0 }).length, 0);
  assert.equal(filterEvents(bundle.events, { ...filters, episodeIds: ['示例 01'] }).length, 2);
});

test('task type selects one detector and ignores obsolete viewer modes', () => {
  const bundle = demoBundle();
  const filters = { side: 'all', frame: 'base', type: 'gripper',
    zMin: -2, zMax: 3, excludeInitial: true, ignoreSeconds: 1 };
  assert.equal(filterEvents(bundle.events, filters).length, 96);
  assert.equal(filterEvents(bundle.events, { ...filters, candidateMode: 'union' }).length, 96);
  assert.equal(filterEvents(bundle.events, { ...filters, eventTypes: ['grasp'] }).length, 48);
  assert.equal(filterEvents(bundle.events, { ...filters, eventTypes: [] }).length, 0);
  assert.equal(filterEvents(bundle.events, { ...filters, eventTypes: ['dwell'] }).length, 0);
  assert.equal(filterEvents(bundle.events, { ...filters, episodeIds: ['示例 01', '示例 02'] }).length, 8);
  assert.equal(filterEvents(bundle.events, { ...filters, episodeIds: [] }).length, 0);
  assert.throws(() => filterEvents(bundle.events, { ...filters, type: 'union' }), /type/);
  assert.equal(operationType(bundle), 'gripper');
  bundle.metadata.detector.motion_detector = 'press';
  assert.equal(operationType(bundle), 'press'); // old published bundle
  bundle.metadata.task_config = { type: 'gripper' };
  assert.equal(operationType(bundle), 'gripper'); // explicit config is authoritative
  bundle.metadata.type = 'press';
  assert.throws(() => operationType(bundle), /不一致/);
  bundle.metadata.type = 'union';
  assert.throws(() => validateBundle(bundle), /type/);
});

test('startup/time exclusions are reversible, per event and stable across episode timestamp offsets', () => {
  const events = demoBundle().events;
  const filters = { episode: 'all', side: 'all', frame: 'base',
    zMin: -2, zMax: 3, type: 'gripper', excludeInitial: true, ignoreSeconds: 1 };
  assert.equal(filterEvents(events, filters).length, 96);
  const audit = filterEvents(events, { ...filters, includeExcluded: true });
  assert.equal(audit.length, 144);
  assert.equal(audit.filter((e) => exclusionReasons(e, filters).length).length, 48);
  assert.equal(filterEvents(events, { ...filters, excludeInitial: false, ignoreSeconds: 0 }).length, 144);
  assert.equal(filterEvents(events, { ...filters, ignoreSeconds: 3 }).length, 48);
  const shifted = events.map((e) => ({ ...e, timestamp_s: e.timestamp_s + 250 }));
  assert.equal(filterEvents(shifted, { ...filters, ignoreSeconds: 3 }).length, 48);
});

test('world to base uses inverse rotation and translation for body snapshots', () => {
  const q = Math.sqrt(.5);
  const snapshot = { body_poses_world: [[1, 2, 0, q, 0, 0, q], [1, 3, .5, q, 0, 0, q]] };
  const matrices = snapshotPoses(snapshot, { body_names: ['base', 'tcp'], base_body: 'base' }, 'base');
  const point = new Vector3().setFromMatrixPosition(matrices[1]);
  assert(point.distanceTo(new Vector3(1, 0, .5)) < 1e-10);
  assert(new Vector3().setFromMatrixPosition(matrices[0]).length() < 1e-10);
  assert(eventPosition({ tcp_pose_base: [1, 2, 3], tcp_pose_world: [4, 5, 6] }, 'world')[0] === 4);
  assert(new Vector3(1, 0, 0).applyMatrix4(poseMatrix(snapshot.body_poses_world[0]))
    .distanceTo(new Vector3(1, 3, 0)) < 1e-10);
});

test('reject wrong units, missing bodies, duplicate IDs and mismatched frames', () => {
  const bad = (mutate, pattern) => {
    const bundle = demoBundle();
    mutate(bundle);
    assert.throws(() => validateBundle(bundle), pattern);
  };
  bad((b) => { b.metadata.units = 'millimeters'; }, /米制/);
  bad((b) => { b.reference_snapshot.body_poses_world.pop(); }, /快照/);
  bad((b) => { b.events[1].event_id = b.events[0].event_id; }, /重复/);
  bad((b) => { b.events[0].tcp_pose_base[0] += 1; }, /坐标不一致/);
  bad((b) => { b.events[0].tcp_pose_world[0] = Infinity; }, /有限数值/);
  const empty = demoBundle();
  empty.events = [];
  assert.equal(validateBundle(empty).events.length, 0);
  bad((b) => { delete b.events[0].detection_sources; }, /来源/);
  bad((b) => { delete b.events[0].initial_phase; }, /来源/);
  const legacy = demoBundle();
  legacy.schema_version = 1;
  legacy.events = legacy.events.filter((e) => e.event_type !== 'dwell');
  legacy.events.forEach((e) => { delete e.detection_sources; delete e.initial_phase; });
  assert.equal(validateBundle(legacy).schema_version, 1);
});

test('press schema selects only press candidates and preserves displacement evidence', () => {
  const bundle = demoBundle();
  bundle.schema_version = 3;
  bundle.events = bundle.events.filter((e) => e.event_type === 'dwell' && !e.initial_phase).slice(0, 3)
    .map((e) => ({ ...e, event_type: 'press', related_event_ids: [],
      press_depth_m: .021, press_retract_m: .016, press_lateral_m: .008 }));
  validateBundle(bundle);
  const filters = { side: 'all', frame: 'base', zMin: -2, zMax: 3 };
  assert.equal(filterEvents(bundle.events, { ...filters, type: 'press' }).length, 3);
  assert.equal(filterEvents(bundle.events, { ...filters, type: 'gripper' }).length, 0);
  delete bundle.events[0].press_depth_m;
  assert.throws(() => validateBundle(bundle), /位移依据/);
});

test('schema 4 objects coexist with historical schemas without reading image references', () => {
  const bundle = demoBundle();
  bundle.schema_version = 4;
  bundle.events.forEach((event) => { event.object = null; event.identification = { status: 'not_requested' }; });
  bundle.events[0].object = '红色方块';
  bundle.events[0].identification = { status: 'success', screenshot: 'screenshots/frame.png' };
  assert.equal(validateBundle(bundle).events[0].object, '红色方块');
  bundle.events[0].identification.screenshot = '../private.png';
  assert.equal(validateBundle(bundle).events[0].object, '红色方块'); // unused image reference
  bundle.events[0].identification.screenshot = 'screenshots/frame.png';
  bundle.events[0].object = 1;
  assert.throws(() => validateBundle(bundle), /物体/);
});

test('schema 1 upgraded by identification retains original candidate filtering contract', () => {
  const bundle = demoBundle();
  bundle.schema_version = 4;
  bundle.metadata.source_schema_version = 1;
  bundle.events = bundle.events.filter((event) => event.event_type === 'grasp');
  bundle.events.forEach((event) => {
    delete event.initial_phase;
    delete event.detection_sources;
    event.object = null;
    event.identification = { status: 'failed' };
  });
  assert.equal(validateBundle(bundle).metadata.source_schema_version, 1);
});
