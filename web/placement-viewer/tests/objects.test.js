import test from 'node:test';
import assert from 'node:assert/strict';
import { Color, Group } from 'three';
import { filterEvents, objectCatalog, objectName, UNKNOWN_OBJECT_COLOR } from '../src/data.js';
import { PlacementScene } from '../src/scene.js';

const events = ['红色方块', '黄色方块', '蓝色方块', '笔帽', '笔筒', null].map((object, i) => ({
  event_id: String(i), episode_id: 'first', object, side: 'left', event_type: 'grasp',
  tcp_pose_base: [i / 10, 0, .8, 1, 0, 0, 0], related_event_ids: [],
}));
const filters = { frame: 'base', side: 'all', zMin: 0, zMax: 2 };

test('object catalog has stable, distinct colors, natural block colors and unknown fallback', () => {
  const catalog = objectCatalog(events);
  assert.deepEqual(objectCatalog([...events].reverse()), catalog);
  assert.equal(new Set(catalog.map((object) => new Color(object.color).getHex())).size, catalog.length);
  assert.equal(catalog.find((object) => object.name === null).color, UNKNOWN_OBJECT_COLOR);
  assert.equal(catalog.find((object) => object.name === '红色方块').color, '#ec6b65');
  assert.deepEqual(objectCatalog([{ object: null }, {}, { object: '  ' }]), [{ name: null, label: '未识别', color: UNKNOWN_OBJECT_COLOR }]);
  assert.equal(objectName({ object: ' 笔帽 ' }), '笔帽');
});

test('object filtering combines with existing filters and respects an empty selection', () => {
  assert.equal(filterEvents(events, filters).length, 6);
  assert.deepEqual(filterEvents(events, { ...filters, objects: ['蓝色方块', null] }).map(objectName), ['蓝色方块', null]);
  assert.equal(filterEvents(events, { ...filters, objects: [] }).length, 0);
  assert.equal(filterEvents(events, { ...filters, objects: ['红色方块'], side: 'right' }).length, 0);
  assert.equal(filterEvents(events, { ...filters, objects: ['红色方块'], zMax: .1 }).length, 0);
  const related = [events[0], { ...events[0], event_id: 'motion', event_type: 'press', object: '蓝色方块', related_event_ids: ['0'] }];
  assert.equal(filterEvents(related, { ...filters, type: 'press', objects: ['蓝色方块'] })[0].event_id, 'motion');
});

test('actual Three.js marker instances use object colors consistently after filtering', () => {
  const scene = Object.create(PlacementScene.prototype);
  scene.markers = new Group();
  scene.density = new Group();
  scene.element = { dataset: {} };
  const colors = new Map(objectCatalog(events).map(({ name, color }) => [name, color]));
  const check = (selected, excluded = new Set()) => {
    scene.renderEvents(selected, 'base', 'points', selected, excluded, .05, colors);
    scene.markers.children.forEach((mesh) => mesh.userData.events.forEach((event, i) => {
      const color = new Color();
      mesh.getColorAt(i, color);
      assert.equal(color.getHex(), new Color(excluded.has(event.event_id) ? '#85858b' : colors.get(objectName(event))).getHex());
    }));
    assert.equal(scene.element.dataset.markerCount, String(selected.length));
  };
  check(events);
  check([events[1], events[4]]);
  check(events, new Set(['0']));
  check([]);
  assert.equal(scene.markers.children.length, 0);
});
