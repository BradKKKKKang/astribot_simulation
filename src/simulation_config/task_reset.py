"""Seedable task-owned placement sampling, without simulator imports."""
import copy
import math

import numpy as np

from .experiment_config import check_fields
from .glb_geometry import rotation_matrix
from .scene_assets import (resolve_scene_objects, transformed_primitives,
                           validate_reset_xy_bounds, validate_reset_xy_samples)


def validate_task_reset(scene, reset):
    if reset is None:
        return
    check_fields(reset, {'objects', 'placement'}, 'task.reset')
    names = reset.get('objects')
    if (not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names)
            or len(names) != len(set(names))):
        raise ValueError('task.reset.objects must be unique object names')
    placement = reset.get('placement', {})
    check_fields(placement, {'type', 'surface', 'min_separation', 'max_attempts'}, 'task.reset.placement')
    if placement.get('type') not in ('uniform_on_surface', 'uniform_in_bounds', 'empirical'):
        raise ValueError('task.reset requires uniform_on_surface, uniform_in_bounds or empirical placement')
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
        if obj.get('asset', {}).get('physics', {}).get('body_type') != 'dynamic':
            raise ValueError('task.reset objects must be dynamic scene assets')
        if obj.get('placement', {}).get('frame') != surface or obj['placement'].get('grounding') != 'geometry_min':
            raise ValueError('task.reset objects must use geometry_min on the selected surface')
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
    """Return sampled bodies; retain orientations, full support and clear footprints.

    Sampling is bounded and completes before any simulator state is written.
    Unselected objects on the same support act as fixed footprint obstacles.
    Calibrated bounds constrain object origins, not the enclosing geometry.
    Empirical reset draws one shared row per attempt, retaining observation
    density and object pairing; rejected rows are redrawn without clipping.
    """
    validate_task_reset(scene, reset)
    sampled = copy.deepcopy(scene)
    names, config = reset['objects'], reset['placement']
    empirical = config['type'] == 'empirical'
    samples = ({name: np.array(scene['objects'][name]['placement']['reset_xy_samples'])
                for name in names} if empirical else {})
    owner, surface_name = config['surface'].split('.', 1)
    support = scene['objects'][owner]
    surface = support['asset']['surfaces'][surface_name]
    bounds = {k: np.array(surface['usable_bounds'][k]) * np.array(support.get('scale', [1, 1, 1]))[:2]
              for k in ('min', 'max')}
    footprints = {}
    for name, obj in scene['objects'].items():
        if obj.get('placement', {}).get('frame') != config['surface'] or 'asset' not in obj:
            continue
        vertices = np.concatenate([p['vertices'] for p in transformed_primitives(obj['asset'], obj.get('scale', [1, 1, 1]))])
        vertices = vertices @ rotation_matrix(obj['placement'].get('quaternion_wxyz', [1, 0, 0, 0])).T
        footprints[name] = (vertices[:, :2].min(0), vertices[:, :2].max(0))
    limits = {}
    for name in names:
        low, high = footprints[name]
        minimum, maximum = bounds['min'] - low, bounds['max'] - high
        if config['type'] in ('uniform_in_bounds', 'empirical'):
            calibrated = scene['objects'][name]['placement']['reset_xy_bounds']
            minimum = np.maximum(minimum, calibrated['min'])
            maximum = np.minimum(maximum, calibrated['max'])
        if np.any(minimum > maximum):
            raise ValueError(f'{name} has no supported placement within the reset range')
        limits[name] = (minimum, maximum)
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
            low, high = footprints[name]
            minimum, maximum = limits[name]
            xy = samples[name][row] if empirical else rng.uniform(minimum, maximum)
            if np.any(xy < minimum) or np.any(xy > maximum):
                break
            box = (low + xy, high + xy)
            if any(np.all(box[0] < other[1] + margin) and np.all(other[0] < box[1] + margin)
                   for other in occupied):
                break
            occupied.append(box)
            placement = sampled['objects'][name]['placement']
            placement.pop('position', None)
            placement['xy'] = xy.tolist()
        else:
            objects, _ = resolve_scene_objects(sampled)
            return {obj['name']: obj for obj in objects if obj['name'] in names}
    raise ValueError('task.reset could not sample nonoverlapping placements within max_attempts')
