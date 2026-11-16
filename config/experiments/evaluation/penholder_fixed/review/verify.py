#!/usr/bin/env python3
"""Capture the fixed layouts through the integrated launcher, or build photo overlays.

Run capture in the project's Isaac/ROS environment with ROS_LOCALHOST_ONLY=1 and
ROS_DOMAIN_ID=198. Report mode needs only NumPy, OpenCV and the project CPU helpers.
Neither mode edits configurations, photographs, meshes or robot poses.
"""
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


def capture(output):
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1' or os.environ.get('ROS_DOMAIN_ID') != '198':
        raise RuntimeError('Capture requires localhost-only ROS domain 198')
    from src.astribot_envs.astribot_envs_factory import AstribotEnvsFactory
    from src.simulation_config.experiment_config import resolve_experiment
    from src.simulation_config.scene_profile import build_experiment_simulation_config
    paths = sorted(CONFIGS.glob('[0-9][0-9]_*.json'))
    output.mkdir(parents=True, exist_ok=False)

    def loop(factory):
        import cv2
        env = factory.astribot_simu_env.unwrapped
        records = {}
        for path in paths:
            resolved = resolve_experiment(path)
            data = build_experiment_simulation_config(resolved)
            # These layouts share robot, task, table and assets. Only dynamic
            # object placement differs; use the normal reset path after loading it.
            if records:
                for section in ('robot', 'task', 'robot_placement'):
                    if resolved['fingerprints'][section] != next(iter(records.values()))['config_fingerprints'][section]:
                        raise ValueError('Batch capture requires identical ' + section)
            for spec in data['scene_objects']:
                if spec['name'] in env.scene_object_keys['dynamic']:
                    body = env.scene[env.scene_object_keys['dynamic'][spec['name']]]
                    body.data.default_root_state[:, :7] = body.data.default_root_state.new_tensor(spec['pose_world'])
            env.reset()
            for _ in range(12):
                env.sim.render()
                for name in env.camera_names:
                    env.scene[name].update(0., force_recompute=True)
            cam = env.scene['head_rgbd']
            rgb = cam.data.output['rgb'][0].cpu().numpy()[:, :, :3]
            if not cv2.imwrite(str(output / (path.stem + '.png')), rgb[:, :, ::-1]):
                raise RuntimeError('Failed to save render')
            records[path.stem] = {
                'config_fingerprints': resolved['fingerprints'],
                'objects': {n: env.scene[k].data.root_state_w[0, :7].cpu().tolist()
                            for n, k in env.scene_object_keys['dynamic'].items()},
                'camera': {'position': cam.data.pos_w[0].cpu().tolist(),
                           'quaternion_wxyz_opencv': cam.data.quat_w_ros[0].cpu().tolist(),
                           'K': cam.data.intrinsic_matrices[0].cpu().tolist()},
                'scene_specs': data['scene_objects'],
                'boundary': 'normal reset; 12 render updates; no physics steps',
            }
            (output / 'capture.json').write_text(json.dumps(records, indent=2) + '\n')
            print('CAPTURED', path.stem, flush=True)
        factory.stop_event.set()

    AstribotEnvsFactory.simu_env_loop = loop
    import astribot_simulation_integrate
    astribot_simulation_integrate.main([
        '--config', str(paths[0]), '--headless', '--no-viewer',
        '--output-dir', str(output / 'evaluation')])


def report(output, photos):
    import cv2
    from src.simulation_config.glb_geometry import rotation_matrix
    from src.simulation_config.scene_assets import transformed_primitives
    from src.simulation_config.experiment_config import resolve_experiment
    annotations = json.loads((CONFIGS / 'review/annotations.json').read_text())
    captures = {phase: json.loads((output / phase / 'capture.json').read_text())
                for phase in ('before', 'after')}
    reviews = output / 'comparisons'
    reviews.mkdir(exist_ok=True)
    rows, metrics = [], []
    for row in annotations['layouts']:
        stem = Path(row['config']).stem
        current = resolve_experiment(CONFIGS / row['config'])
        for section in ('scene', 'robot', 'task', 'robot_placement'):
            if current['fingerprints'][section] != captures['after'][stem]['config_fingerprints'][section]:
                raise ValueError('Stale after capture: ' + stem + ' / ' + section)
        photo = photos / row['photo']
        original = cv2.imread(str(photo))
        if original is None:
            raise FileNotFoundError(photo)
        x0, y0, x1, y1 = row['viewport_xyxy']
        reference = cv2.resize(original[y0:y1, x0:x1], (1280, 720))
        def normalize(points):
            return (np.asarray(points) - [x0, y0]) * [1280 / (x1 - x0), 720 / (y1 - y0)]
        foreground = np.zeros((720, 1280), np.uint8)
        polygon = row['occlusion']['foreground_polygon_px']
        annotated = reference.copy()
        if polygon:
            p = np.round(normalize(polygon)).astype(np.int32)
            cv2.fillPoly(foreground, [p], 255)
            cv2.polylines(annotated, [p], True, (255, 0, 255), 2)
            cv2.putText(annotated, 'MAGENTA: photo occluder annotation (approximate)',
                        (18, 697), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 255), 2)
        cv2.imwrite(str(reviews / (stem + '_reference.png')), reference)
        cv2.imwrite(str(reviews / (stem + '_occluder.png')), foreground)
        cv2.imwrite(str(reviews / (stem + '_annotated.png')), annotated)
        stat = {'config': row['config'], 'photo_sha256': hashlib.sha256(photo.read_bytes()).hexdigest(),
                'occlusion': row['occlusion'], 'simulation_comparison': row['simulation_comparison'], 'phases': {}}
        overlays = []
        for phase in ('before', 'after'):
            c = captures[phase][stem]
            cam = c['camera']
            R = rotation_matrix(cam['quaternion_wxyz_opencv'])
            K = np.asarray(cam['K'])
            def project(points):
                camera_points = (points - cam['position']) @ R
                pixels = camera_points @ K.T
                return pixels[:, :2] / pixels[:, 2:]
            pose = np.asarray(c['objects']['pen'])
            pen_spec = next(s for s in c['scene_specs'] if s['name'] == 'pen')
            pen_vertices = np.concatenate([p['vertices'] for p in transformed_primitives(
                pen_spec['asset'], pen_spec['scale'])])
            endpoints = np.array([[0, 0, pen_vertices[:, 2].min()],
                                  [0, 0, pen_vertices[:, 2].max()]])
            tips = project(endpoints @ rotation_matrix(pose[3:]).T + pose[:3])
            observed = row['pen_endpoints_cap_rear_px']
            errors = [float(np.linalg.norm(tips[i] - normalize(point)))
                      for i, point in enumerate(observed) if point is not None]
            spec = next(s for s in c['scene_specs'] if s['name'] == 'penholder')
            verts = np.concatenate([p['vertices'] for p in transformed_primitives(spec['asset'], spec['scale'])])
            hp = np.asarray(c['objects']['penholder'])
            hpixels = project(verts @ rotation_matrix(hp[3:]).T + hp[:3])
            center = (hpixels.min(0) + hpixels.max(0)) / 2
            target = normalize(np.asarray(row['holder_bbox_xyxy_px']).reshape(2, 2)).mean(0)
            stat['phases'][phase] = {
                'pen_observed_endpoint_errors_px': errors,
                'pen_projected_endpoints_px': tips.tolist(),
                'holder_projected_bbox_center_error_px': float(np.linalg.norm(center - target)),
                'holder_projected_size_px': np.ptp(hpixels, axis=0).tolist(),
            }
            render = cv2.imread(str(output / phase / (stem + '.png')))
            if render is None or render.shape != reference.shape:
                raise ValueError('Missing or invalid render: ' + stem)
            overlay = cv2.addWeighted(reference, .5, render, .5, 0)
            # Keep raw overlays unmasked: do not conceal simulation occlusion errors.
            cv2.imwrite(str(reviews / (stem + '_' + phase + '_overlay.png')), overlay)
            overlays.append(overlay)
        raw = cv2.imread(str(output / 'after' / (stem + '.png')))
        panels = []
        for label, pic in zip(('PHOTO / OCCLUSION', 'BEFORE 50% OVERLAY', 'AFTER 50% OVERLAY', 'AFTER RAW SIM'),
                              (annotated, *overlays, raw)):
            pic = cv2.resize(pic, (480, 270))
            panel = np.full((302, 480, 3), 30, np.uint8)
            panel[32:] = pic
            cv2.putText(panel, stem[:2] + ' ' + label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
        metrics.append(stat)
    for start in (0, 5):
        cv2.imwrite(str(output / f'contact_sheet_{start+1:02}_{start+5:02}.jpg'), np.concatenate(rows[start:start+5], axis=0))
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    options = ''.join(f'<option value="{Path(r["config"]).stem}">{r["config"]}</option>' for r in annotations['layouts'])
    metadata = json.dumps({Path(r['config']).stem: r['simulation_comparison'] for r in annotations['layouts']})
    html = '''<!doctype html><meta charset="utf-8"><title>Penholder photo overlay review</title>
<style>body{font:16px sans-serif;background:#191c20;color:#eee;margin:24px}select,input{margin:10px}#stack{position:relative;max-width:1280px;aspect-ratio:16/9}#stack img{position:absolute;inset:0;width:100%}p{max-width:1100px}a{color:#82cfff}</style>
<h1>Penholder photo overlay review</h1><p>The placement fit adjusted object XY and pen yaw. Current pen: 160 mm long, 20 mm maximum diameter; original baseline: 150 x 16 mm. Robot pose is unchanged. Photo crops remove viewer chrome; no perspective warp. Raw overlays retain simulator occlusion differences. Endpoint/center residuals measure projection fit to manual annotations, not metric calibration.</p>
<label>Layout <select id="layout">OPTIONS</select></label><label>Render <select id="phase"><option>after</option><option>before</option></select></label><label>Simulation opacity <input id="alpha" type="range" min="0" max="100" value="50"></label><label><input id="occ" type="checkbox">Show photo occluder annotation</label>
<p id="note"></p><div id="stack"><img id="photo"><img id="sim"></div><p>Known differences: photo 01's barrel is hidden in reality but exposed in the simulator; 05 has extra cap occlusion in simulation; 06/08 have different overlap boundaries; 07's pen is closer to the finger silhouette; 09's holder–gripper overlap is not reproduced. Holder size/shape and robot/camera alignment are not calibrated here. Original photo occlusion annotations are preserved separately; no foreground mask is applied to the simulation.</p><p><a href="metrics.json">Projection metrics and source hashes</a> · <a href="contact_sheet_01_05.jpg">Layouts 01–05</a> · <a href="contact_sheet_06_10.jpg">Layouts 06–10</a></p>
<script>const notes=METADATA;function update(){const s=document.querySelector('#layout').value,p=document.querySelector('#phase').value;document.querySelector('#photo').src='comparisons/'+s+(document.querySelector('#occ').checked?'_annotated.png':'_reference.png');document.querySelector('#sim').src=p+'/'+s+'.png';document.querySelector('#sim').style.opacity=document.querySelector('#alpha').value/100;document.querySelector('#note').textContent=notes[s];}document.querySelectorAll('input,select').forEach(x=>x.addEventListener('input',update));update();</script>'''
    (output / 'index.html').write_text(html.replace('OPTIONS', options).replace('METADATA', metadata))
    print('Wrote', output / 'index.html')
    for row in metrics:
        print(row['config'], {p: {'pen_max_px': round(max(v['pen_observed_endpoint_errors_px']), 2),
              'holder_center_px': round(v['holder_projected_bbox_center_error_px'], 2)} for p, v in row['phases'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'report'])
    parser.add_argument('--output', type=Path, required=True,
                        help='capture: new directory; report: parent containing before/ and after/')
    parser.add_argument('--photos', type=Path, default=Path.home() / 'Downloads')
    args = parser.parse_args()
    if args.mode == 'capture':
        capture(args.output.resolve())
    else:
        report(args.output.resolve(), args.photos)
