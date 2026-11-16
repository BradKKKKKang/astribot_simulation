#!/usr/bin/env python3
"""Validate fixed layouts, capture Isaac resets, and compare with source photos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
CONFIGS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def layouts():
    from src.evaluation.managed import validate_managed_config
    from src.simulation_config.experiment_config import resolve_experiment
    from src.simulation_config.scene_profile import build_experiment_simulation_config
    from src.simulation_config.scene_assets import transformed_primitives
    annotations = json.loads((CONFIGS / 'review/annotations.json').read_text())
    paths = sorted(CONFIGS.glob('[0-9][0-9].json'))
    assert [p.stem for p in paths] == [f'{i:02}' for i in range(1, 11)]
    result = []
    for path in paths:
        resolved = resolve_experiment(path)
        data = build_experiment_simulation_config(resolved)
        validate_managed_config(resolved, data)
        sections = resolved['sections']
        assert sections['runtime']['evaluation'] == {'episodes': 1, 'max_steps': 6500, 'seed': 0}
        assert sections['task']['reset'] is None
        specs = {s['name']: s for s in data['scene_objects']}
        assert len(specs) == 8
        bear_physics = specs['bear']['asset']['physics']
        assert bear_physics['body_type'] == 'deformable', path.name
        assert bear_physics['collision'] == 'tetrahedral', path.name
        assert bear_physics['friction'] == 2.0, path.name
        assert bear_physics['mass'] == .015, path.name
        assert bear_physics['deformable']['youngs_modulus'] == 30000, path.name
        assert bear_physics['deformable']['simulation_hexahedral_resolution'] == 20, path.name
        pig_physics = specs['pig']['asset']['physics']
        assert pig_physics['body_type'] == 'dynamic', path.name
        assert pig_physics['collision'] == 'convex_hull', path.name
        assert pig_physics['friction'] == 2.0, path.name
        assert pig_physics['mass'] == .015, path.name
        for name, spec in specs.items():
            if name != 'table':
                assert abs(spec['bounds_world']['min'][2] - .751) < 1e-7
        if result:
            first = result[0][1]
            for key in ('robot', 'task', 'robot_placement'):
                assert resolved['fingerprints'][key] == first['fingerprints'][key], key
            first_specs = {s['name']: s for s in result[0][2]['scene_objects']}
            assert specs['table'] == first_specs['table']
            for name, spec in specs.items():
                assert spec['asset'] == first_specs[name]['asset']
                assert spec['scale'] == first_specs[name]['scale']
        else:
            pig = specs['pig']
            pig_primitives = transformed_primitives(pig['asset'], pig['scale'])
            assert not any('_Loops_' in p['node'] for p in pig_primitives)
            assert sum(len(p['faces']) for p in pig_primitives) == 106838
            bear = specs['bear']
            primitives = transformed_primitives(bear['asset'], bear['scale'])
            points = np.concatenate([p['vertices'] for p in primitives])
            head = np.concatenate([p['vertices'] for p in primitives
                                   if p['node'].startswith(('Head_', 'Ear_', 'Nose_', 'HangingLoop_'))])
            face = np.concatenate([p['vertices'] for p in primitives if p['node'].startswith('Head_')])
            torso = np.concatenate([p['vertices'] for p in primitives if p['node'].startswith('Torso_')])
            assert not any(p['node'].endswith('_ShortPlushPile') for p in primitives)
            assert sum(len(p['faces']) for p in primitives) == 113280
            # Blue torso narrows by 1 cm; limbs move inward, and ears now
            # determine the whole bear's maximum width.
            assert abs(np.ptp(points[:, 0]) - .06693060337258223) < 1e-7
            assert abs(np.ptp(torso[:, 0]) - .032657366520862985) < 1e-7
            assert abs(np.ptp(face[:, 0]) - .04879222291078442) < 1e-7
            # Source +90 X rotation maps glTF depth Z to asset Y.
            assert abs(np.ptp(face[:, 1]) - .05918472644863808) < 1e-7
            assert abs(np.ptp(head[:, 1]) - .06462589709922112) < 1e-7
            for name, placement in annotations['baseline_placements'].items():
                assert sections['scene']['objects'][name]['placement'] == placement
        result.append((path, resolved, data))
    assert len({r['sections']['output']['root'] for _, r, _ in result}) == 10
    assert len({r['fingerprints']['scene'] for _, r, _ in result}) == 10
    return result


def capture(output):
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '197':
        raise RuntimeError('Capture requires ROS_LOCALHOST_ONLY=1 and ROS_DOMAIN_ID=197')
    configurations = layouts()
    output.mkdir(parents=True, exist_ok=False)
    from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory

    def loop(factory):
        import cv2
        from src.simulation_config.isaac_scene_objects import reset_scene_objects, read_scene_object_states
        env = factory.astribot_simu_env.unwrapped
        records = {}
        for path, resolved, data in configurations:
            # Validated above: identical static table, robot, task and assets.
            env.reset(seed=42)
            sampled = {s['name']: s for s in data['scene_objects']
                       if s['asset']['physics']['body_type'] != 'static'}
            # Use the shared reset adapter for both rigid and FEM bodies.
            reset_scene_objects(env.scene, env.scene_object_keys, sampled=sampled)
            states = read_scene_object_states(env.scene, env.scene_object_keys)
            reset_objects = {name: states[name]['pose_world'] for name in sampled}
            for name, actual in reset_objects.items():
                expected = sampled[name]['pose_world']
                assert np.allclose(actual[:3], expected[:3], atol=1e-6)
                assert abs(abs(np.dot(actual[3:], expected[3:])) - 1) < 1e-5
            # PhysX refreshes the FEM render skin on the next physics tick.
            post_reset_steps = int(bool(env.scene_object_keys.get('deformable')))
            if post_reset_steps:
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.sim.get_physics_dt())
            for _ in range(32):
                env.sim.render()
                for name in env.camera_names:
                    env.scene[name].update(0., force_recompute=True)
                    _ = env.scene[name].data.output['rgb']
            folder = output / path.stem
            folder.mkdir()
            cameras = {}
            for name in env.camera_names:
                cam = env.scene[name]
                rgb = env.render_single_camera(name)['rgb_img'][:, :, :3]
                assert cv2.imwrite(str(folder / (name + '.png')), rgb[:, :, ::-1])
                distortion = env._camera_distortions.get(name)
                cameras[name] = {
                    'position': cam.data.pos_w[0].cpu().tolist(),
                    'quaternion_wxyz_opencv': cam.data.quat_w_ros[0].cpu().tolist(),
                    'K': cam.data.intrinsic_matrices[0].cpu().tolist(),
                    'distortion': distortion.coefficients.tolist() if distortion else [0.] * 5,
                }
            states = read_scene_object_states(env.scene, env.scene_object_keys)
            objects = {name: states[name]['pose_world'] for name in sampled}
            assert all(np.isfinite(pose).all() for pose in objects.values())
            records[path.stem] = {
                'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'fingerprints': resolved['fingerprints'], 'scene_specs': data['scene_objects'],
                'objects': objects, 'reset_objects': reset_objects, 'cameras': cameras,
                'seed': 42, 'post_reset_physics_steps': post_reset_steps,
                'boundary': 'integrated launcher; shared rigid/FEM reset; FEM skin refresh; 32 render updates; normal lens distortion',
            }
            (output / 'capture.json').write_text(json.dumps(records, indent=2) + '\n')
            print('CAPTURED', path.stem, flush=True)
        factory.stop_event.set()

    AstribotEnvsFactory.simu_env_loop = loop
    import astribot_simulation_integrate
    astribot_simulation_integrate.main([
        '--config', str(configurations[0][0]), '--headless', '--no-viewer',
        '--output-dir', str(output / 'evaluation')])


def report(output, photos):
    import cv2
    configurations = layouts()
    records = json.loads((output / 'capture.json').read_text())
    annotations = json.loads((CONFIGS / 'review/annotations.json').read_text())
    rows = []
    for (path, resolved, _), annotation in zip(configurations, annotations['layouts']):
        record = records[path.stem]
        assert record['config_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record['fingerprints'] == resolved['fingerprints']
        raw = cv2.imread(str(output / path.stem / 'head_rgbd.png'))
        assert raw is not None and raw.shape == (720, 1280, 3)
        if annotation['photo']:
            photo = photos / Path(annotation['photo']).name
            assert hashlib.sha256(photo.read_bytes()).hexdigest() == annotation['photo_sha256']
            original = cv2.imread(str(photo))
            x0, y0, x1, y1 = annotation['viewport_xyxy']
            reference = cv2.resize(original[y0:y1, x0:x1], (1280, 720))
            overlay = cv2.addWeighted(reference, .5, raw, .5, 0)
            cv2.imwrite(str(output / path.stem / 'reference.png'), reference)
            cv2.imwrite(str(output / path.stem / 'overlay.png'), overlay)
        else:
            reference = cv2.imread(str(ROOT / 'outputs/evaluation/tabletop_objects/aligned_120x60/head_rgbd.png'))
            assert reference is not None
            overlay = cv2.addWeighted(reference, .5, raw, .5, 0)
        panels = []
        for label, pic in zip(('REFERENCE' if path.stem != '01' else 'PREVIOUS SIM', 'FIXED SIM', '50% OVERLAY'), (reference, raw, overlay)):
            panel = np.full((302, 480, 3), 26, np.uint8)
            panel[32:] = cv2.resize(pic, (480, 270))
            cv2.putText(panel, path.stem + ' ' + label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    for start in (0, 5):
        cv2.imwrite(str(output / f'comparison_{start+1:02}_{start+5:02}.jpg'), np.concatenate(rows[start:start+5], axis=0))
    summary = {'layouts': 10, 'max_steps': 6500, 'baseline_01_preserved': True,
               'managed_admission': 'passed', 'ground_clearance_m': .001,
               'table_footprint_containment': 'passed for actual mesh vertices',
               'capture_config_hashes': 'matched', 'reset_pose_verification': 'passed',
               'post_reset_physics_steps': sorted({r['post_reset_physics_steps'] for r in records.values()}),
               'limits': 'Initial placement only. Hidden object extents are estimates; no dynamics or task-success validation.'}
    (output / 'validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'capture', 'report'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--photos', type=Path, default=Path.home() / 'Downloads')
    args = parser.parse_args()
    if args.mode == 'check':
        for path, _, _ in layouts():
            print(path.name, 'OK')
    elif args.output is None:
        parser.error('--output is required for capture/report')
    elif args.mode == 'capture':
        capture(args.output.resolve())
    else:
        report(args.output.resolve(), args.photos)
