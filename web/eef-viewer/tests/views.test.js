import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { selectCameraSlots } from '../src/video-group.js';
import { alignRobotToChassis, referenceDescription } from '../src/view-geometry.js';
import { EefScene } from '../src/scene.js';

test('stereo head names never replace left/right wrist views', () => {
  const cameras = ['head_left','head_right','left','right'].map(name => ({camera:`observation.images.${name}`}));
  const selected = selectCameraSlots(cameras, 'observation.images.head_right');
  assert.equal(selected.head.camera, 'observation.images.head_right');
  assert.equal(selected.left.camera, 'observation.images.left');
  assert.equal(selected.right.camera, 'observation.images.right');
  assert.equal(selectCameraSlots(cameras.slice(0, 2)).left, null);
});
test('head order in the manifest does not change the three-view assignment', () => {
  const cameras = ['right','left','head'].map(name => ({camera:`observation.images.${name}`}));
  assert.equal(selectCameraSlots(cameras).head.camera, 'observation.images.head');
});
test('Franka base cameras supply the main video without becoming wrist cameras', () => {
  const cameras = ['right','base_view_right','base_view_front'].map(name => ({camera:`observation.images.${name}`}));
  const selected = selectCameraSlots(cameras);
  assert.equal(selected.head.camera, 'observation.images.base_view_front');
  assert.equal(selected.right.camera, 'observation.images.right');
  assert.equal(selected.left, null);
  assert.equal(selected.heads.length, 2);
  assert.equal(selectCameraSlots(cameras, 'observation.images.base_view_right').head.camera,
    'observation.images.base_view_right');
});
test('one-camera and external-only Franka recordings remain playable', () => {
  const camera = {camera:'observation.images.base_view_front'};
  assert.deepEqual(selectCameraSlots([camera]), {head:camera,left:null,right:null,heads:[camera]});
  const wrist = {camera:'observation.images.right'};
  assert.deepEqual(selectCameraSlots([wrist]), {head:wrist,left:null,right:null,heads:[wrist]});
  const external = ['base_view_left','base_view_right'].map(name => ({camera:`observation.images.${name}`}));
  const selected = selectCameraSlots(external);
  assert.ok(selected.head);
  assert.equal(selected.left, null);
  assert.equal(selected.right, null);
});
test('USD model and SDK targets share a chassis origin independently of the ground height', () => {
  const world = new THREE.Group(), robot = new THREE.Group();
  const base = new THREE.Mesh(new THREE.BoxGeometry(.4, .4, .2), new THREE.MeshBasicMaterial());
  base.position.z = -.15; robot.add(base); world.add(robot);
  const chassis = new THREE.Object3D(), tcp = new THREE.Object3D();
  chassis.position.z = -.097; tcp.position.set(.4, .2, .8 - .097);
  robot.add(chassis, tcp);
  const result = alignRobotToChassis(robot, [0, 0, -.097]);
  assert.ok(chassis.getWorldPosition(new THREE.Vector3()).length() < 1e-12);
  assert.ok(tcp.getWorldPosition(new THREE.Vector3()).distanceTo(new THREE.Vector3(.4, .2, .8)) < 1e-12);
  assert.ok(Math.abs(result.groundZ - (-.25 + .097)) < 1e-8);
  assert.ok(Math.abs(new THREE.Box3().setFromObject(robot, true).min.z - result.groundZ) < 1e-12);
  assert.deepEqual(alignRobotToChassis(robot, [0, 0, -.097]), result);
  assert.equal(base.position.z, -.15);
  base.geometry.dispose(); base.material.dispose();
});
test('state, action, achieved EEF, residuals and trails all stay in chassis coordinates', () => {
  const scene = Object.create(EefScene.prototype);
  scene.targets = new THREE.Group(); scene.trails = new THREE.Group(); scene.requestRender = () => {};
  scene.pose = () => {};
  const target = [.5, .2, .8, 0, 0, 0, 1], achieved = [.4, .2, .8, 0, 0, 0, 1];
  const action = [.6, .3, .9, 0, 0, 0, 1];
  const frame = {state: {targets: {left: target}}, action: {targets: {left: action}}};
  scene.showFrame(frame, {achieved: {left: achieved}, ik_success: true}, 'state', true, true);
  const [stateMarker, actionMarker, achievedMarker, residual] = scene.targets.children;
  assert.deepEqual(stateMarker.position.toArray(), target.slice(0, 3));
  assert.deepEqual(actionMarker.position.toArray(), action.slice(0, 3));
  assert.deepEqual(achievedMarker.position.toArray(), achieved.slice(0, 3));
  const positions = residual.geometry.getAttribute('position');
  for (const [i, pose] of [target, achieved].entries()) {
    assert.ok(new THREE.Vector3().fromBufferAttribute(positions, i).distanceTo(new THREE.Vector3(...pose.slice(0, 3))) < 1e-7);
  }
  scene.trajectory([frame, {state: {targets: {left: action}}}], 'state', true);
  const trail = scene.trails.children[0].geometry.getAttribute('position');
  assert.ok(Math.abs(trail.getZ(0) - target[2]) < 1e-7);
  assert.ok(Math.abs(trail.getZ(1) - action[2]) < 1e-7);
  scene.clear(scene.targets); scene.clear(scene.trails);
});
test('corrected G2 frame does not apply the original source rotation again', () => {
  const result = referenceDescription({review:{status:'admitted'},coordinates:{target_frame:'rotated_source',
    resolved_base_matrix:[[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]}});
  assert.equal(result.name, 'rotated_source');
  assert.deepEqual(result.dataToChassis, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
  assert.deepEqual(result.chassisOffset, [0,0,-.097]);
  assert.equal(result.showData, true);
  assert.equal(referenceDescription({review:{status:'deferred'}}).showData, false);
});
