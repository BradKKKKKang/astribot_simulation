import test from 'node:test';
import assert from 'node:assert/strict';
import { densityField } from '../src/field.js';
import { eventPosition, filterEvents } from '../src/data.js';
import { demoBundle } from '../src/demo.js';

const event = (x, y = 0, z = 0) => ({ tcp_pose_base: [x, y, z], tcp_pose_world: [x + 2, y, z] });
const nearest = (field, position) => {
  const [x, y, z] = position.map((v, i) => Math.round((v - field.min[i]) / field.step));
  return field.values[x + field.dimensions[0] * (y + field.dimensions[1] * z)];
};

test('Gaussian volume peaks at actual 3D samples and decays in every axis', () => {
  const field = densityField([event(0)], 'base', .05);
  assert(Math.abs(nearest(field, [0, 0, 0]) - 1) < 1e-6);
  for (let axis = 0; axis < 3; axis++) {
    const position = [0, 0, 0];
    position[axis] = .05;
    assert(Math.abs(nearest(field, position) - Math.exp(-.5)) < 1e-5);
    position[axis] = -.05;
    assert(Math.abs(nearest(field, position) - Math.exp(-.5)) < 1e-5);
  }
  assert(field.min[2] < 0 && field.size[2] > .25); // a volume, not XY projection
});

test('density weights repeated events, preserves metric frames and bounds memory', () => {
  const events = [event(0), event(0), event(0), event(1)];
  const base = densityField(events, 'base', .05);
  assert(nearest(base, [0, 0, 0]) > nearest(base, [1, 0, 0]) * 2.9);
  const world = densityField(events, 'world', .05);
  assert(Math.abs(world.min[0] - base.min[0] - 2) < 1e-10);
  assert(Math.abs(world.size[0] - base.size[0]) < 1e-10);
  assert(base.values.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
  const wide = densityField([event(-100, -100, -100), event(100, 100, 100)], 'base', .02);
  assert(wide === null || wide.values.length <= 64 ** 3);
});

test('empty density is absent and smoothing scale has physical units', () => {
  assert.equal(densityField([], 'base', .05), null);
  assert.throws(() => densityField([event(0)], 'base', 0), /正数/);
  const narrow = densityField([event(0)], 'base', .025);
  const wide = densityField([event(0)], 'base', .1);
  assert(Math.abs(wide.size[0] / narrow.size[0] - 4) < 1e-10);
});

test('demo motion hotspots follow episode ranges 1–24, 23–24 and 24', () => {
  const bundle = demoBundle();
  bundle.events = bundle.events.filter((e) => e.event_type === 'dwell').map((e) => ({ ...e, event_type: 'press' }));
  const episodes = [...new Set(bundle.events.map((e) => e.episode_id))].sort();
  const stages = [1, 23, 24].map((start) => {
    const events = filterEvents(bundle.events, { episodeIds: episodes.slice(start - 1),
      type: 'press', side: 'left', frame: 'base', zMin: -2, zMax: 3,
      ignoreSeconds: 1, excludeInitial: true });
    const field = densityField(events, 'base', .05);
    const index = field.values.indexOf(1);
    const grid = [index % field.dimensions[0],
      Math.floor(index / field.dimensions[0]) % field.dimensions[1],
      Math.floor(index / (field.dimensions[0] * field.dimensions[1]))];
    return { events, field, peak: grid.map((v, i) => field.min[i] + v * field.step) };
  });
  assert.deepEqual(stages.map(({ events }) => events.length), [36, 3, 1]);
  for (let i = 1; i < stages.length; i++) {
    const distance = Math.hypot(...stages[i].peak.map((v, a) => v - stages[i - 1].peak[a]));
    assert(distance > stages[i].field.step, 'remaining motion samples must move the demo hotspot');
  }
  const last = stages.at(-1);
  assert(nearest(last.field, eventPosition(last.events[0], 'base')) > .99);
  assert(nearest(last.field, stages[0].peak) < .1, 'removed episodes must not retain their hotspot');
});

test('compact dense cluster has highest local density even against a deeper sparse column', () => {
  const dense = Array.from({ length: 12 }, () => event(-.3));
  const column = [-.4, -.2, 0, .2, .4].flatMap((z) => Array.from({ length: 4 }, () => event(.3, 0, z)));
  const field = densityField([...dense, ...column], 'base', .05);
  const center = nearest(field, [-.3, 0, 0]);
  assert(Math.abs(center - 1) < 1e-6);
  for (const z of [-.4, -.2, 0, .2, .4]) assert(nearest(field, [.3, 0, z]) < center * .35);
  // Validate the whole volume's highest sample stays inside the dense cluster.
  const index = field.values.indexOf(Math.max(...field.values));
  const x = index % field.dimensions[0];
  const y = Math.floor(index / field.dimensions[0]) % field.dimensions[1];
  const z = Math.floor(index / (field.dimensions[0] * field.dimensions[1]));
  assert(Math.hypot(...[x, y, z].map((v, axis) => field.min[axis] + v * field.step - [-.3, 0, 0][axis])) < field.bandwidth / 2);
});
