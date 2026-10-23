import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { eventPosition, snapshotPoses } from './data.js';
import { densityField, fieldVertexShader, fieldFragmentShader } from './field.js';

const COLORS = { grasp: 0x21c5b5, release: 0xf4ab62, dwell: 0x9c9cfa, press: 0x9c9cfa };

export class PlacementScene {
  constructor(element, onSelect) {
    this.element = element;
    this.onSelect = onSelect;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14212c);
    this.camera = new THREE.PerspectiveCamera(39, 1, 0.01, 200);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(2.5, -2.9, 2);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.3;
    element.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0.35, 0, 0.8);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 100;
    this.scene.add(new THREE.HemisphereLight(0xe1f4ff, 0x475060, 2.5));
    const key = new THREE.DirectionalLight(0xfff1df, 3);
    key.position.set(2, -3, 6);
    const fill = new THREE.DirectionalLight(0x9bcde6, 1.4);
    fill.position.set(-2, 3, 2);
    this.scene.add(key, fill);
    this.grid = new THREE.GridHelper(6, 60, 0x58717d, 0x314550);
    this.grid.rotation.x = Math.PI / 2;
    this.scene.add(this.grid);
    const axes = new THREE.AxesHelper(0.35);
    axes.position.z = 0.002;
    this.scene.add(axes);
    this.robot = new THREE.Group();
    this.markers = new THREE.Group();
    this.density = new THREE.Group();
    this.scene.add(this.robot, this.markers, this.density);
    this.selected = new THREE.Mesh(new THREE.SphereGeometry(0.022, 20, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, depthTest: false }));
    this.selected.visible = false;
    this.selected.renderOrder = 5;
    this.scene.add(this.selected);
    this.raycaster = new THREE.Raycaster();
    let down = null;
    element.addEventListener('pointerdown', (event) => { down = [event.clientX, event.clientY]; });
    element.addEventListener('pointerup', (event) => {
      if (!down || Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 5) return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      const pointer = new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1,
        -(event.clientY - rect.top) / rect.height * 2 + 1);
      this.raycaster.setFromCamera(pointer, this.camera);
      const hit = this.raycaster.intersectObjects(this.markers.children).find((h) => h.instanceId !== undefined);
      if (hit) this.onSelect(hit.object.userData.events[hit.instanceId]);
    });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(element);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const { width, height } = this.element.getBoundingClientRect();
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  clear(group) {
    const geometries = new Set();
    const materials = new Set();
    group.traverse((node) => {
      if (node.geometry) geometries.add(node.geometry);
      if (node.material) (Array.isArray(node.material) ? node.material : [node.material]).forEach((m) => materials.add(m));
      if (node.isInstancedMesh) node.dispose();
    });
    geometries.forEach((g) => g.dispose());
    materials.forEach((m) => {
      if (m.uniforms?.densityMap) m.uniforms.densityMap.value.dispose();
      m.dispose();
    });
    group.clear();
  }

  async loadRobot(url, metadata, snapshot) {
    const loadId = (this.robotLoadId ?? 0) + 1;
    this.robotLoadId = loadId;
    let next;
    if (!url) throw new Error('必须提供与 Astribot S1 数据匹配的 robot.glb');
    const gltf = await new GLTFLoader().loadAsync(url);
    next = gltf.scene;
    const missing = metadata.body_names.filter((name) => !next.getObjectByName(name));
    if (missing.length) {
      this.clear(next);
      throw new Error(`机器人 GLB 与事件数据不匹配：缺少 ${missing.slice(0, 3).join('、')}`);
    }
    next.traverse((node) => {
      if (!node.isMesh) return;
      if (!node.geometry.attributes.normal) node.geometry.computeVertexNormals();
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      materials.forEach((material) => { material.transparent = true; material.depthWrite = false; material.opacity = 0.65; });
    });
    if (loadId !== this.robotLoadId) {
      this.clear(next);
      return;
    }
    this.clear(this.robot);
    this.robot.add(next);
    this.metadata = metadata;
    this.pose(snapshot, 'base');
  }

  pose(snapshot, frame) {
    if (!snapshot || !this.metadata) return;
    snapshotPoses(snapshot, this.metadata, frame).forEach((matrix, i) => {
      const node = this.robot.getObjectByName(this.metadata.body_names[i]);
      if (node) {
        matrix.decompose(node.position, node.quaternion, node.scale);
        node.updateMatrix();
      }
    });
    this.robot.updateMatrixWorld(true);
  }

  opacity(value) {
    this.robot.traverse((node) => {
      if (!node.material) return;
      (Array.isArray(node.material) ? node.material : [node.material]).forEach((material) => {
        material.opacity = value;
        material.transparent = value < 1;
        material.depthWrite = value >= 1;
        material.needsUpdate = true;
      });
    });
  }

  renderEvents(events, frame, mode, counted = events, excludedIds = new Set(), bandwidth = .05) {
    this.clear(this.markers);
    this.clear(this.density);
    for (const type of ['grasp', 'release', 'dwell', 'press']) {
      const selected = events.filter((event) => event.event_type === type);
      if (!selected.length) continue;
      const geometry = type === 'grasp' ? new THREE.SphereGeometry(0.009, 12, 8)
        : type === 'release' ? new THREE.OctahedronGeometry(0.013) : new THREE.BoxGeometry(0.018, 0.018, 0.018);
      const mesh = new THREE.InstancedMesh(geometry,
        new THREE.MeshBasicMaterial({ color: COLORS[type], depthTest: false }), selected.length);
      selected.forEach((event, index) => {
        mesh.setMatrixAt(index, new THREE.Matrix4().makeTranslation(...eventPosition(event, frame)));
        mesh.setColorAt(index, new THREE.Color(excludedIds.has(event.event_id) ? 0x85858b : COLORS[type]));
      });
      mesh.material.color.set(0xffffff);
      mesh.instanceColor.needsUpdate = true;
      mesh.userData.events = selected;
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
      mesh.renderOrder = 3;
      this.markers.add(mesh);
    }
    const isField = mode.startsWith('field');
    if (isField) {
      const field = densityField(counted, frame, bandwidth);
      if (field) {
        const texture = new THREE.Data3DTexture(field.values, ...field.dimensions);
        texture.format = THREE.RedFormat;
        texture.type = THREE.FloatType;
        texture.minFilter = texture.magFilter = THREE.LinearFilter;
        texture.unpackAlignment = 1;
        texture.needsUpdate = true;
        const material = new THREE.ShaderMaterial({
          glslVersion: THREE.GLSL3,
          vertexShader: fieldVertexShader, fragmentShader: fieldFragmentShader,
          uniforms: { densityMap: { value: texture }, cameraLocal: { value: new THREE.Vector3() },
            dimensions: { value: new THREE.Vector3(...field.dimensions) } },
          side: THREE.BackSide, transparent: true, depthWrite: false, depthTest: false,
        });
        const volume = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
        volume.position.set(...field.min.map((v, i) => v + field.size[i] / 2));
        volume.scale.set(...field.size);
        volume.renderOrder = 1;
        volume.onBeforeRender = () => {
          material.uniforms.cameraLocal.value.copy(volume.worldToLocal(this.camera.getWorldPosition(new THREE.Vector3())));
        };
        this.density.add(volume);
      }
    }
    // Keep markers pickable even when displaying only density, without visible
    // marker materials. They provide the identity of individual events.
    this.markers.children.forEach((mesh) => { mesh.material.visible = mode !== 'field'; });
    this.density.visible = mode !== 'points';
    this.element.dataset.densityKind = isField ? 'field' : 'none';
    this.element.dataset.densitySamples = String(counted.length);
    this.element.dataset.densityMeshes = String(this.density.children.length);
  }

  select(event, frame) {
    this.selected.visible = Boolean(event);
    if (event) this.selected.position.set(...eventPosition(event, frame));
  }

  fit(view = 'perspective') {
    const bounds = new THREE.Box3().setFromObject(this.robot);
    this.markers.children.forEach((mesh) => mesh.userData.events.forEach((event) => {
      bounds.expandByPoint(new THREE.Vector3(...eventPosition(event, this.frame ?? 'base')));
    }));
    if (this.density.visible) bounds.union(new THREE.Box3().setFromObject(this.density));
    if (bounds.isEmpty()) bounds.setFromCenterAndSize(new THREE.Vector3(0.4, 0, 0.8), new THREE.Vector3(2, 2, 2));
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const distance = Math.max(size.x, size.y, size.z, 1) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)))
      / Math.min(this.camera.aspect, 1) * 1.25;
    const direction = { perspective: [1.4, -1.8, 1.05], top: [0, 0.0001, 1],
      front: [1, 0, 0], side: [0, -1, 0] }[view];
    this.camera.position.copy(center).add(new THREE.Vector3(...direction).normalize().multiplyScalar(distance));
    this.controls.target.copy(center);
    this.controls.update();
  }
}
