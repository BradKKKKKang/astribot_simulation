import * as THREE from 'three';
import { PlacementScene } from '../../placement-viewer/src/scene.js';
import { alignRobotToChassis, referenceDescription } from './view-geometry.js';

function point(pose) { return new THREE.Vector3(...pose.slice(0, 3)); }

export class EefScene extends PlacementScene {
  constructor(element) {
    super(element, () => {});
    this.camera.position.set(1.9, -2.1, 1.65);
    // Render on pose/camera changes. A paused review must not continuously
    // rasterize the full reference model, especially on software WebGL clients.
    this.renderer.setAnimationLoop(null);
    this.controls.enableDamping = false;
    this.controls.addEventListener('change', () => this.requestRender());
    this.targets = new THREE.Group(); this.trails = new THREE.Group();
    this.presentation = new THREE.Group();
    this.scene.add(this.presentation);
    this.presentation.add(this.robot, this.targets, this.trails);
    // Use one labeled world/chassis frame instead of the inherited axes.
    for (const child of this.scene.children) if (child.type === 'AxesHelper') child.visible = false;
    this.createReferenceFrame();
    this.controls.update();
    this.requestRender();
  }
  requestRender() {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      this.renderer.render(this.scene, this.camera);
      this.updateFrameLabels();
    });
  }
  resize() { super.resize(); this.requestRender(); }
  pose(snapshot, frame) { super.pose(snapshot, frame); this.requestRender(); }
  async loadRobot(...args) {
    await super.loadRobot(...args);
    super.pose(args[2], 'world');
    this.robot.traverse(node => {
      if (node.isMesh) for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
        material.transparent = false; material.opacity = 1; material.depthWrite = true; material.needsUpdate = true;
      }
    });
    this.alignModel(true);
    this.requestRender();
  }
  alignModel(resetGround = false) {
    if (!this.robot.children.length) return;
    const previous = this.robot.position.clone();
    if (resetGround || !this.groundInfo) {
      this.groundInfo = alignRobotToChassis(this.robot, this.reference.chassisOffset);
    } else {
      this.robot.position.fromArray(this.reference.chassisOffset).negate();
      this.groundInfo.groundZ += this.robot.position.z - previous.z;
    }
    // The grid follows the initial model's lower bound, never the coordinate
    // origin. Keep this height fixed during playback.
    this.grid.position.z = this.groundInfo.groundZ;
    const delta = this.robot.position.clone().sub(previous);
    this.camera.position.add(delta); this.controls.target.add(delta);
    this.controls.update();
    this.element.dataset.groundHeight = String(this.groundInfo.groundZ);
    this.element.dataset.modelTranslation = JSON.stringify(this.robot.position.toArray());
    this.updateFrameNote();
  }
  createReferenceFrame() {
    this.referenceFrame = new THREE.Group();
    this.presentation.add(this.referenceFrame);
    const axes = new THREE.AxesHelper(.40);
    axes.setColors(0xff7777, 0x69da9d, 0x7caeff);
    axes.material.depthTest = false; axes.renderOrder = 1000;
    this.referenceFrame.add(axes);
    const origin = new THREE.Mesh(new THREE.SphereGeometry(.014, 12, 8), new THREE.MeshBasicMaterial({color:0xffffff,depthTest:false}));
    origin.renderOrder = 1002; this.referenceFrame.add(origin);
    const points = [[0,0,0],[.30,0,0],[0,0,0],[0,.30,0],[0,0,0],[0,0,.30]].map(p => new THREE.Vector3(...p));
    this.dataAxes = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineDashedMaterial({color:0xffce79,dashSize:.022,gapSize:.015,depthTest:false}));
    this.dataAxes.computeLineDistances(); this.dataAxes.renderOrder = 1001;
    this.dataAxes.matrixAutoUpdate = false; this.referenceFrame.add(this.dataAxes);
    this.frameLabels = [
      ['X',[.44,0,0],'#ff7777'], ['Y',[0,.44,0],'#69da9d'], ['Z',[0,0,.44],'#7caeff'], ['世界 · SDK chassis',[0,0,0],'#ffe0a7'],
    ].map(([text,position,color],i) => {
      const element = document.createElement('span'); element.className = `frame-label ${i === 3 ? 'frame-origin' : ''}`;
      element.textContent = text; element.style.color = color; this.element.appendChild(element);
      return {element,position:new THREE.Vector3(...position)};
    });
    this.setReferenceFrame(null);
  }
  setReferenceFrame(episode) {
    this.reference = referenceDescription(episode);
    this.referenceFrame.position.set(0, 0, 0);
    this.dataAxes.visible = this.reference.showData;
    this.dataAxes.matrix.set(...this.reference.dataToChassis.flat());
    this.frameLabels[3].element.textContent = this.reference.showData ? '世界 · chassis · 数据系' : '世界 · SDK chassis';
    this.element.dataset.dataFrame = this.reference.name;
    this.alignModel();
    this.updateFrameNote(); this.requestRender();
  }
  setReferenceVisible(visible) {
    this.referenceFrame.visible = visible; this.requestRender();
  }
  updateFrameNote() {
    const note = document.querySelector('#frame-note');
    if (!note || !this.reference) return;
    const relation = this.reference.showData ? `校正数据系：${this.reference.name} · 与 chassis 暂同原点、同轴向（未标定）` : '参考系：SDK chassis';
    note.textContent = `世界原点 = SDK chassis 原点 (0, 0, 0)\n${relation}`;
  }
  updateFrameLabels() {
    if (!this.referenceFrame || !this.frameLabels) return;
    const {width,height} = this.element.getBoundingClientRect();
    this.element.dataset.chassisOrigin = JSON.stringify(this.referenceFrame.getWorldPosition(new THREE.Vector3()).toArray());
    for (const label of this.frameLabels) {
      const projected = this.referenceFrame.localToWorld(label.position.clone()).project(this.camera);
      label.element.hidden = !this.referenceFrame.visible || Math.abs(projected.x) > 1 || Math.abs(projected.y) > 1 || Math.abs(projected.z) > 1;
      label.element.style.left = `${(projected.x + 1) * width / 2}px`;
      label.element.style.top = `${(1 - projected.y) * height / 2}px`;
    }
  }
  showFrame(frame, solved, mode, showAction, metric) {
    this.clear(this.targets);
    if (solved) this.pose({ body_poses_world: solved.body_poses_world }, 'world');
    if (!metric) return;
    for (const [side, color] of [['left', 0x5ce0c0], ['right', 0xb19aff]]) {
      const target = frame[mode].targets[side];
      if (target) this.marker(target, color, .085);
      if (showAction && mode !== 'action' && frame.action.targets[side]) this.marker(frame.action.targets[side], 0xf1a24a, .055);
      if (solved?.achieved?.[side]) {
        const achieved = solved.achieved[side];
        this.marker(achieved, solved.ik_success ? 0xffffff : 0xf3a14d, .04);
        if (target) {
          const geometry = new THREE.BufferGeometry().setFromPoints([point(target), point(achieved)]);
          this.targets.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: solved.ik_success ? color : 0xff6a79 })));
        }
      }
    }
    this.requestRender();
  }
  marker(pose, color, size) {
    const group = new THREE.Group();
    group.position.copy(point(pose)); group.quaternion.set(...pose.slice(3));
    group.add(new THREE.AxesHelper(size));
    group.add(new THREE.Mesh(new THREE.SphereGeometry(size / 9, 10, 8), new THREE.MeshBasicMaterial({ color, depthTest: false })));
    this.targets.add(group);
  }
  trajectory(frames, mode, metric) {
    this.clear(this.trails);
    if (!metric) return;
    for (const [side, color] of [['left', 0x5ce0c0], ['right', 0xb19aff]]) {
      let previous = null; const points = [];
      for (const frame of frames) {
        const pose = frame[mode].targets[side];
        if (pose && previous) points.push(point(previous), point(pose));
        previous = pose;
      }
      const line = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color, transparent: true, opacity: .45 }));
      this.trails.add(line);
    }
    this.requestRender();
  }
}
