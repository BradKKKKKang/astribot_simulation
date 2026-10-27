import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { Group } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { demoBundle } from '../src/demo.js';
import { artifactUrl, expandManifest, snapshotLoader } from '../src/bundle.js';
import { PlacementScene } from '../src/scene.js';

const reference = (value) => {
  const sha256 = createHash('sha256').update(JSON.stringify(value)).digest('hex');
  return { path: `objects/${sha256.slice(0, 2)}/${sha256}.json`, sha256 };
};

test('schema5 loads thin points and reference first, snapshots only when selected and once', async () => {
  const legacy = demoBundle(), calls = [], resources = new Map();
  const put = (value) => { const ref = reference(value); resources.set(ref.sha256, value); return ref; };
  const snapshot = { body_poses_world: legacy.reference_snapshot.body_poses_world.map((p) => [p[0] + 1, ...p.slice(1)]) };
  const event = { ...legacy.events.find((e) => e.event_type === 'grasp'), object: null,
    identification: { status: 'unknown' }, snapshot_ref: put(snapshot) };
  const manifest = { ...legacy, schema_version: 5, events: undefined, artifact_base: '/artifacts/store/',
    display: { event_types: ['grasp'], default_event_types: ['grasp'], event_labels: { grasp: '自定义抓取' } },
    reference_snapshot: put(legacy.reference_snapshot),
    episodes: [{ episode_id: event.episode_id, events: put({ episode_id: event.episode_id, events: [event] }) }] };
  const fetcher = async (url) => { const digest = url.split('/').at(-1).split('.')[0]; calls.push(digest);
    return { ok: true, json: async () => structuredClone(resources.get(digest)) }; };
  const bundle = await expandManifest(manifest, 'http://localhost/bundles/a.json', fetcher);
  assert.equal(calls.length, 2);
  assert(!calls.includes(event.snapshot_ref.sha256));
  const load = snapshotLoader(fetcher);
  assert.deepEqual(await load(bundle.events[0], bundle, 'http://localhost/'), snapshot);
  await load(bundle.events[0], bundle, 'http://localhost/');
  assert.equal(calls.filter((digest) => digest === event.snapshot_ref.sha256).length, 1);
  assert.throws(() => artifactUrl({ ...event.snapshot_ref, path: '../secret' }, '/', 'http://localhost/'));
  assert.throws(() => artifactUrl(event.snapshot_ref, 'http://elsewhere/', 'http://localhost/'));
  const invalid = snapshotLoader(async () => ({ ok: true, json: async () => ({ body_poses_world: [] }) }));
  await assert.rejects(invalid(bundle.events[0], bundle, 'http://localhost/'), /body_names/);
});

test('unchanged model URL reuses loaded geometry across event revisions', async () => {
  const scene = Object.create(PlacementScene.prototype);
  scene.robot = new Group();
  const body = new Group(); body.name = 'base';
  const geometry = new Group(); geometry.add(body); scene.robot.add(geometry);
  scene.robotUrl = 'http://localhost/artifacts/store/model.glb';
  scene.metadata = { body_names: ['base'], base_body: 'base' };
  const snapshot = { body_poses_world: [[1, 2, 3, 1, 0, 0, 0]] };
  const old = GLTFLoader.prototype.loadAsync;
  GLTFLoader.prototype.loadAsync = async () => { throw new Error('must not reload GLB'); };
  try {
    await scene.loadRobot(scene.robotUrl, { ...scene.metadata, revision: 2 }, snapshot);
    assert.equal(scene.robot.children[0], geometry);
    assert.equal(scene.metadata.revision, 2);
  } finally { GLTFLoader.prototype.loadAsync = old; }
});

test('schema5 view filters honor backend analysis selection and never reinterpret exclusion thresholds', async () => {
  const { filterEvents, exclusionReasons } = await import('../src/data.js');
  const event = { event_id: 'one', episode_id: 'one', side: 'left', event_type: 'grasp',
    tcp_pose_base: [0, 0, 1], initial_phase: true, episode_elapsed_s: 0,
    analysis_selected: true, analysis_exclusion_reasons: [] };
  const filters = { frame: 'base', side: 'all', zMin: 0, zMax: 2, analysisSelection: true,
    excludeInitial: true, ignoreSeconds: 100 };
  assert.equal(filterEvents([event], filters).length, 1);
  assert.deepEqual(exclusionReasons(event, filters), []);
  const excluded = { ...event, analysis_selected: false, analysis_exclusion_reasons: ['initial_pose'] };
  assert.equal(filterEvents([excluded], filters).length, 0);
  assert.equal(filterEvents([excluded], { ...filters, includeExcluded: true }).length, 1);
  assert.deepEqual(exclusionReasons(excluded, { ...filters, excludeInitial: false, ignoreSeconds: 0 }), ['initial_pose']);
});
