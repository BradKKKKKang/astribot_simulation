"""Material tracking for the controlled cloth sweep; no simulator or ROS imports."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.cloth_grasp_diagnostics import compile_finger_hulls, _plane_clearance
from src.simulation_config.glb_geometry import rotation_matrix


def analyze(directory):
    status = json.loads((directory/'run_status.json').read_text())
    if status['status'] != 'completed' or status['server_exit_code'] != 0:
        raise RuntimeError('trial did not complete cleanly: '+directory.name)
    trace = Path(status['trace'])
    states = {}
    for line in trace.open():
        row = json.loads(line)
        if row['event'] == 'state': states[row['step_id']] = row
    assert set(states) == set(range(1231)), 'missing physics boundaries'
    geometry = json.loads((trace.parent/'collision_geometry.json').read_text())
    fingers = ['astribot_gripper_right_Link_'+s for s in ('L11','R11')]
    hulls = {f:compile_finger_hulls(geometry['bodies'][f]) for f in fingers}
    def body(a, name): return a['body_poses'][a['body_names'].tolist().index(name)]
    def selected(a):
        mask = np.ones(len(a['points']), bool)
        for f in fingers:
            p = body(a,f)
            mask &= _plane_clearance((a['points']-p[:3])@rotation_matrix(p[3:]), hulls[f], .006) <= .006
        return np.flatnonzero(mask)
    anchor = np.load(directory/'particles/000930.npz')
    approach = np.load(directory/'particles/000850.npz')
    ids = selected(anchor)
    wrist0 = body(anchor,'astribot_arm_right_link_7')
    centers = []
    for f in fingers:
        pose = body(approach,f)
        verts = np.concatenate([h['vertices'] for h in hulls[f]])
        centers.append(verts.mean(0)@rotation_matrix(pose[3:]).T+pose[:3])
    center = np.mean(centers,axis=0)
    material_center = approach['points'][np.argmin(np.linalg.norm(approach['points']-center,axis=1))]
    patch = np.flatnonzero(np.linalg.norm(approach['points']-material_center,axis=1) <= .015)
    assert len(patch), 'no local material patch'
    metrics = []
    for path in sorted((directory/'particles').glob('*.npz')):
        step = int(path.stem)
        if step < 930: continue
        a = np.load(path)
        p = body(a,'astribot_arm_right_link_7')
        near = selected(a)
        geo = states[step]['diagnostics']['cloth_proximity']['tshirt']['right']
        rel = (a['points'][patch]-p[:3])@rotation_matrix(p[3:])
        rel0 = (anchor['points'][patch]-wrist0[:3])@rotation_matrix(wrist0[3:])
        metrics.append({'step':step, 'time_s':step*.01, 'gap_mm':geo['closing_axis_gap_m']*1000,
            'wrist_delta_mm':((p[:3]-wrist0[:3])*1000).tolist(),
            'rotation_deg':float(np.rad2deg(Rotation.from_matrix(rotation_matrix(wrist0[3:]).T@rotation_matrix(p[3:])).magnitude())),
            'bilateral_candidates':len(near), 'retained_candidates':len(np.intersect1d(ids,near)),
            'candidate_lift_mm':float(np.median(a['points'][ids,2]-anchor['points'][ids,2])*1000) if len(ids) else None,
            'patch_lift_mm':float(np.median(a['points'][patch,2]-anchor['points'][patch,2])*1000),
            'patch_slip_mm':float(np.median(np.linalg.norm(rel-rel0,axis=1))*1000),
            'inside_hull_particles':geo['inside_hull_particles']})
    end = metrics[-1]
    drift = max(np.linalg.norm(r['wrist_delta_mm'][:2]) for r in metrics)
    rotation = max(r['rotation_deg'] for r in metrics)
    valid = abs(end['wrist_delta_mm'][2]-20) <= 1 and drift <= 2 and rotation <= 1
    result = {'case':directory.name, 'condition':status['condition'], 'motion_valid':bool(valid),
        'closed_gap_mm':metrics[0]['gap_mm'], 'wrist_lift_mm':end['wrist_delta_mm'][2],
        'horizontal_drift_mm':float(drift), 'rotation_deg':rotation,
        'initial_candidates':len(ids), 'retained_candidates':end['retained_candidates'],
        'candidate_lift_mm':end['candidate_lift_mm'], 'patch_lift_mm':end['patch_lift_mm'], 'patch_slip_mm':end['patch_slip_mm'],
        'tracked_ids':ids.tolist(), 'patch_ids':patch.tolist(), 'video':str(directory/'interaction.mp4'),
        'note':'Candidates use the 6 mm contact band at step 930. Patch is selected BEFORE closure at step 850, within 15 mm of the nearest material point to the cooked finger hull centroid. Lift is relative to step 930. IDs are fixed within each trial; closure groups share the same pre-intervention particles, while translated cloth can change welding. Geometric proximity is not force or task success.'}
    (directory/'analysis.json').write_text(json.dumps(result,indent=2))
    (directory/'metrics.json').write_text(json.dumps(metrics,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    summaries=[]
    for case in json.loads((args.output/'manifest.json').read_text())['conditions']:
        directory=args.output/case['name']
        status=directory/'run_status.json'
        if args.partial and (not status.exists() or json.loads(status.read_text())['status']!='completed'): continue
        result=analyze(directory)
        summaries.append(result)
        print({k:result[k] for k in ('case','closed_gap_mm','wrist_lift_mm','patch_lift_mm','retained_candidates','motion_valid')},flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries,indent=2))
    baseline_path=args.output/'baseline/particles/000850.npz'
    if baseline_path.exists():
        baseline=np.load(baseline_path)['points']
        initial_baseline=np.load(args.output/'baseline/particles/000000.npz')['points']
        controls=[]
        for result in summaries:
            initial=np.load(args.output/result['case']/'particles/000000.npz')['points']
            translation=np.array([*result['condition']['shirt_world_xy_m'],0.])
            initial_equal=initial.shape==initial_baseline.shape
            record={'case':result['case'],'particle_count':len(initial),
                'initial_translation_residual_m':float(np.max(np.abs(initial-translation-initial_baseline))) if initial_equal else None}
            if result['condition']['shirt_world_xy_m'] != [0.,0.]:
                controls.append(record)
                continue
            points=np.load(args.output/result['case']/'particles/000850.npz')['points']
            same_shape=points.shape==baseline.shape
            controls.append({**record,'pre_intervention_particles_identical':bool(same_shape and np.array_equal(points,baseline)),
                'maximum_particle_difference_m':float(np.max(np.abs(points-baseline))) if same_shape else None,
                'fixed_patch_ids_equal_to_baseline':result['patch_ids']==summaries[0]['patch_ids']})
        (args.output/'control_validation.json').write_text(json.dumps(controls,indent=2))
    fields=['case','motion_valid','closed_gap_mm','wrist_lift_mm','initial_candidates','retained_candidates','candidate_lift_mm','patch_lift_mm','patch_slip_mm']
    with (args.output/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(summaries)

if __name__=='__main__': main()
