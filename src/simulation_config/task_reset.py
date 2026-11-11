"""Seedable task-owned placement sampling, without simulator imports."""
import copy
import math

import numpy as np

from .experiment_config import check_fields
from .glb_geometry import rotation_matrix
from .poses import quaternion_multiply
from .scene_assets import (resolve_scene_objects, transformed_primitives,
                           validate_reset_xy_bounds, validate_reset_xy_samples,
                           validate_reset_yaw_range_degrees)


def validate_task_reset(scene, reset):
    if reset is None:
        return
    check_fields(reset, {'objects', 'placement', 'settle_steps'}, 'task.reset')
    names = reset.get('objects')
    if (not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names)
            or len(names) != len(set(names))):
        raise ValueError('task.reset.objects must be unique object names')
    placement = reset.get('placement', {})
    check_fields(placement, {'type', 'surface', 'min_separation', 'max_attempts', 'regions'}, 'task.reset.placement')
    if placement.get('type') not in ('uniform_on_surface', 'uniform_in_bounds', 'empirical', 'uniform_in_regions'):
        raise ValueError('task.reset requires uniform_on_surface, uniform_in_bounds or empirical placement')
    steps=reset.get('settle_steps',0)
    if type(steps) is not int or not 0<=steps<=5000:
        raise ValueError('task.reset.settle_steps must be an integer in [0,5000]')
    if placement['type']=='uniform_in_regions':
        regions=placement.get('regions')
        if len(names)!=1 or not isinstance(regions,list) or not regions:
            raise ValueError('uniform_in_regions requires one object and nonempty regions')
        ids=[]
        for region in regions:
            check_fields(region,{'id','min','max'},'reset region')
            if not isinstance(region.get('id'),str) or not region['id']: raise ValueError('reset region id required')
            validate_reset_xy_bounds({k:region[k] for k in ('min','max')})
            ids.append(region['id'])
        if len(ids)!=len(set(ids)): raise ValueError('reset region ids must be unique')
    elif 'regions' in placement:
        raise ValueError('reset regions require uniform_in_regions')
    surface = placement.get('surface')
    if not isinstance(surface, str) or '.' not in surface:
        raise ValueError('task.reset requires an asset support surface')
    owner, surface_name = surface.split('.', 1)
    support = scene.get('objects', {}).get(owner, {}).get('asset', {})
    if support.get('physics', {}).get('body_type') != 'static' or surface_name not in support.get('surfaces', {}):
        raise ValueError('task.reset requires a static asset support surface')
    sample_counts = set()
    for name in names:
        obj = scene.get('objects', {}).get(name, {})
        if obj.get('asset', {}).get('physics', {}).get('body_type') not in ('dynamic', 'deformable', 'cloth'):
            raise ValueError('task.reset objects must be dynamic or deformable scene assets')
        if obj.get('placement', {}).get('frame') != surface or obj['placement'].get('grounding') != 'geometry_min':
            raise ValueError('task.reset objects must use geometry_min on the selected surface')
        if 'reset_yaw_range_degrees' in obj['placement']:
            validate_reset_yaw_range_degrees(obj['placement']['reset_yaw_range_degrees'])
        bounds = obj['placement'].get('reset_xy_bounds')
        if placement['type'] in ('uniform_in_bounds', 'empirical'):
            if bounds is None:
                raise ValueError(f'{name}: {placement["type"]} requires placement.reset_xy_bounds')
            low, high = validate_reset_xy_bounds(bounds)
        elif bounds is not None:
            raise ValueError(f'{name}: reset_xy_bounds requires uniform_in_bounds or empirical placement')
        samples = obj['placement'].get('reset_xy_samples')
        if placement['type'] == 'empirical':
            points = validate_reset_xy_samples(samples)
            if np.any(points < low) or np.any(points > high):
                raise ValueError(f'{name}: reset_xy_samples extends beyond reset_xy_bounds')
            sample_counts.add(len(points))
        elif samples is not None:
            raise ValueError(f'{name}: reset_xy_samples requires empirical placement')
    if len(sample_counts) > 1:
        raise ValueError('empirical reset requires equal sample counts for paired objects')
    margin = placement.get('min_separation', 0.)
    if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
        raise ValueError('task.reset min_separation must be finite and nonnegative')
    attempts = placement.get('max_attempts', 100)
    if type(attempts) is not int or attempts < 1:
        raise ValueError('task.reset max_attempts must be a positive integer')


def sample_task_objects(scene, reset, rng):
    """Return sampled bodies with full support and clear rotated footprints.

    Sampling is bounded and completes before any simulator state is written.
    Unselected objects on the same support act as fixed footprint obstacles.
    Calibrated bounds constrain object origins, not the enclosing geometry.
    Empirical reset draws one shared row per attempt, retaining observation
    density and object pairing; rejected rows are redrawn without clipping.
    Optional yaw offsets rotate about support-frame Z from the base orientation.
    """
    validate_task_reset(scene, reset)
    sampled = copy.deepcopy(scene)
    names, config = reset['objects'], reset['placement']
    region = config['regions'][int(rng.integers(len(config['regions'])))] if config['type']=='uniform_in_regions' else None
    empirical = config['type'] == 'empirical'
    samples = ({name: np.array(scene['objects'][name]['placement']['reset_xy_samples'])
                for name in names} if empirical else {})
    owner, surface_name = config['surface'].split('.', 1)
    support = scene['objects'][owner]
    surface = support['asset']['surfaces'][surface_name]
    bounds = {k: np.array(surface['usable_bounds'][k]) * np.array(support.get('scale', [1, 1, 1]))[:2]
              for k in ('min', 'max')}
    footprints, base_vertices = {}, {}
    for name, obj in scene['objects'].items():
        if obj.get('placement', {}).get('frame') != config['surface'] or 'asset' not in obj:
            continue
        vertices = np.concatenate([p['vertices'] for p in transformed_primitives(obj['asset'], obj.get('scale', [1, 1, 1]))])
        vertices = vertices @ rotation_matrix(obj['placement'].get('quaternion_wxyz', [1, 0, 0, 0])).T
        base_vertices[name] = vertices
        footprints[name] = (vertices[:, :2].min(0), vertices[:, :2].max(0))
    occupied = []
    for name, (low, high) in footprints.items():
        if name not in names:
            placement = scene['objects'][name]['placement']
            xy = np.array(placement.get('xy', placement.get('position', [0, 0, 0])[:2]))
            occupied.append((low + xy, high + xy))
    margin = config.get('min_separation', 0.)
    fixed = list(occupied)
    for _ in range(config.get('max_attempts', 100)):
        occupied = list(fixed)
        row = int(rng.integers(len(samples[names[0]]))) if empirical else None
        for name in names:
            original = scene['objects'][name]['placement']
            placement = sampled['objects'][name]['placement']
            low, high = footprints[name]
            yaw_range = original.get('reset_yaw_range_degrees')
            if yaw_range is not None:
                half_angle = math.radians(rng.uniform(*yaw_range)) / 2
                yaw = [math.cos(half_angle), 0, 0, math.sin(half_angle)]
                placement['quaternion_wxyz'] = quaternion_multiply(
                    yaw, original.get('quaternion_wxyz', [1, 0, 0, 0])).tolist()
                rotated = base_vertices[name] @ rotation_matrix(yaw).T
                low, high = rotated[:, :2].min(0), rotated[:, :2].max(0)
            minimum, maximum = bounds['min'] - low, bounds['max'] - high
            if config['type'] in ('uniform_in_bounds', 'empirical'):
                calibrated = original['reset_xy_bounds']
                minimum = np.maximum(minimum, calibrated['min'])
                maximum = np.minimum(maximum, calibrated['max'])
            if region is not None:
                minimum=np.maximum(minimum,region['min'])
                maximum=np.minimum(maximum,region['max'])
            if np.any(minimum > maximum):
                if yaw_range is None:
                    raise ValueError(f'{name} has no supported placement within the reset range')
                break
            xy = samples[name][row] if empirical else rng.uniform(minimum, maximum)
            if np.any(xy < minimum) or np.any(xy > maximum):
                break
            box = (low + xy, high + xy)
            if any(np.all(box[0] < other[1] + margin) and np.all(other[0] < box[1] + margin)
                   for other in occupied):
                break
            occupied.append(box)
            placement.pop('position', None)
            placement['xy'] = xy.tolist()
        else:
            objects, _ = resolve_scene_objects(sampled)
            result={obj['name']: obj for obj in objects if obj['name'] in names}
            if region is not None:
                result[names[0]]['reset_region']=region['id']
            return result
    raise ValueError('task.reset could not sample nonoverlapping placements within max_attempts')
