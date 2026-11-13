import test from 'node:test';
import assert from 'node:assert/strict';
import { nearestFrame, nextAnomaly } from '../src/timeline.js';

test('uses irregular timestamps instead of a fixed frame rate', () => {
  const times = [0, .033, .1, .243, .5];
  assert.equal(nearestFrame(times, .23), 3);
  assert.equal(nearestFrame(times, .39), 4);
  assert.equal(nearestFrame(times, -1), 0);
  assert.equal(nearestFrame(times, 3), 4);
  assert.equal(nearestFrame([], 1), -1);
});
test('next anomaly wraps and handles no failed frames', () => {
  assert.equal(nextAnomaly([3, 9, 12], 9), 12);
  assert.equal(nextAnomaly([3, 9, 12], 12), 3);
  assert.equal(nextAnomaly([], 8), 8);
});
