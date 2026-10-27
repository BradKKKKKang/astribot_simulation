"""Asset profiles and geometry-derived support placement; CPU-only configuration."""
import copy
from functools import lru_cache
import math
from pathlib import Path
import re

import numpy as np

from .glb_geometry import read_glb, rotation_matrix
from .isaac_scene_config import pose_fields, vector, _fields
from .poses import compose_pose


@lru_cache(maxsize=16)
def _geometry(path, modified, size):
    return read_glb(path)


def asset_geometry(asset):
    path = Path(asset['source']['path']).expanduser().resolve(strict=True)
    stat = path.stat()
    geometry = _geometry(str(path), stat.st_mtime_ns, stat.st_size)
    if geometry['sha256'] != asset['source']['sha256']:
        raise ValueError(f'asset source digest changed: {asset["name"]}')
    return geometry


def transformed_primitives(asset, scale=(1., 1., 1.)):
    transform = asset['source_transform']
    matrix = np.diag(scale) @ rotation_matrix(transform['quaternion_wxyz']) @ np.diag(transform['scale'])
    result = []
    for original in asset_geometry(asset)['primitives']:
        item = {**original, 'vertices': original['vertices'] @ matrix.T,
                'attributes': dict(original['attributes'])}
        if 'NORMAL' in item['attributes']:
            normals = item['attributes']['NORMAL'] @ np.linalg.inv(matrix)
            item['attributes']['NORMAL'] = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        result.append(item)
    return result


def validate_asset(asset):
    _fields(asset, {'name', 'source', 'source_transform', 'physics', 'surfaces'}, 'asset')
    if not isinstance(asset.get('name'), str) or not asset['name']:
        raise ValueError('asset.name must be nonempty')
    source = asset.get('source', {})
    _fields(source, {'path', 'sha256', 'description'}, 'asset.source')
    if not isinstance(source.get('path'), str) or not source['path'] or '$' in source['path']:
        raise ValueError('asset.source.path requires a literal GLB path')
    if not re.fullmatch('[0-9a-f]{64}', source.get('sha256', '')):
        raise ValueError('asset.source.sha256 requires the source content digest')
    transform = asset.get('source_transform', {})
    _fields(transform, {'quaternion_wxyz', 'scale'}, 'asset.source_transform')
    if set(transform) != {'quaternion_wxyz', 'scale'}:
        raise ValueError('asset source axis rotation and scale must be explicit')
    pose_fields(transform, 'asset.source_transform')
    if any(v <= 0 for v in vector(transform['scale'], 3, 'asset.source_transform.scale')):
        raise ValueError('asset scale must be positive')
    physics = asset.get('physics', {})
    _fields(physics, {'body_type', 'collision', 'mass', 'friction', 'restitution'}, 'asset.physics')
    if physics.get('body_type') not in ('static', 'dynamic'):
        raise ValueError('asset body_type must be static or dynamic')
    if physics.get('collision') not in ('triangle_mesh', 'convex_hull', 'sdf'):
        raise ValueError('unsupported asset collision representation')
    if physics['body_type'] == 'dynamic' and physics['collision'] == 'triangle_mesh':
        raise ValueError('dynamic concave assets require sdf collision')
    for field in ('mass', 'friction', 'restitution'):
        if field in physics and (type(physics[field]) not in (int, float) or not math.isfinite(physics[field])
                                 or physics[field] < 0 or field == 'mass' and physics[field] == 0):
            raise ValueError(f'asset.physics.{field} must be finite and nonnegative (mass positive)')
    if physics['body_type'] == 'dynamic' and 'mass' not in physics:
        raise ValueError('dynamic asset requires explicit mass')
    surfaces = asset.get('surfaces', {})
    if not isinstance(surfaces, dict):
        raise ValueError('asset.surfaces must be an object')
    primitives = transformed_primitives(asset)
    for name, surface in surfaces.items():
        _fields(surface, {'position', 'quaternion_wxyz', 'normal', 'long_direction', 'short_direction',
                          'usable_bounds', 'geometry_node', 'description'}, f'asset.surfaces.{name}')
        position, quaternion = pose_fields(surface, f'surface.{name}')
        basis = rotation_matrix(quaternion)
        for key, axis in (('normal', 2), ('long_direction', 0), ('short_direction', 1)):
            if not np.allclose(vector(surface.get(key), 3, key), basis[:, axis], atol=1e-7):
                raise ValueError('surface normal and edge directions must match its frame')
        bounds = surface.get('usable_bounds', {})
        _fields(bounds, {'min', 'max'}, 'surface.usable_bounds')
        low, high = (np.array(vector(bounds.get(k), 2, 'surface bounds')) for k in ('min', 'max'))
        if np.any(low >= high):
            raise ValueError('surface bounds require positive area')
        selected = [p['vertices'] for p in primitives if p['node'] == surface.get('geometry_node')]
        if not selected:
            raise ValueError('surface geometry_node does not exist in source')
        points = (np.concatenate(selected) - position) @ basis
        top = points[np.isclose(points[:, 2], points[:, 2].max(), atol=1e-6)]
        if abs(points[:, 2].max()) > 1e-6 or not np.allclose((top.min(0) + top.max(0))[:2] / 2, 0, atol=1e-6):
            raise ValueError('surface origin must be the selected geometry top-face center')
        if np.any(low < top[:, :2].min(0)) or np.any(high > top[:, :2].max(0)):
            raise ValueError('surface usable bounds exceed its top face')
        triangles = np.concatenate([p['vertices'][p['faces']] for p in primitives
                                    if p['node'] == surface['geometry_node']])
        triangles = (triangles - position) @ basis
        triangles = triangles[np.all(np.abs(triangles[:, :, 2]) < 1e-6, axis=1), :, :2]
        # Rounded panel bounds include unsupported corner space. Check the
        # usable rectangle against actual top-face triangles, not only the AABB.
        for corner in ((low[0], low[1]), (low[0], high[1]), (high[0], low[1]), (high[0], high[1])):
            a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            point = np.array(corner) - triangles[:, 0]
            determinant = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
            valid = np.abs(determinant) > 1e-12
            u = np.divide(point[:, 0]*b[:, 1] - point[:, 1]*b[:, 0], determinant,
                          out=np.full_like(determinant, -1.), where=valid)
            v = np.divide(a[:, 0]*point[:, 1] - a[:, 1]*point[:, 0], determinant,
                          out=np.full_like(determinant, -1.), where=valid)
            if not np.any(valid & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1+1e-7)):
                raise ValueError('surface usable rectangle corners require supporting top-face geometry')
    return asset


def _asset_profile(spec, directory, seen=()):
    from .experiment_config import _merge, check_fields, read_json
    if not isinstance(spec, dict):
        raise ValueError('asset must be a profile or explicit reference')
    if 'ref' in spec:
        check_fields(spec, {'ref', 'overrides'}, 'asset reference')
        if not isinstance(spec['ref'], str) or not spec['ref'] or '$' in spec['ref']:
            raise ValueError('asset ref must be a literal path')
        path = (Path(directory) / spec['ref']).expanduser().resolve(strict=True)
        if path in seen:
            raise ValueError('cyclic asset profile reference')
        base = _asset_profile(read_json(path), path.parent, (*seen, path))
        override = _asset_profile(spec.get('overrides', {}), directory, seen)
        return _merge(base, override)
    asset = copy.deepcopy(spec)
    source = asset.get('source', {})
    if isinstance(source.get('path'), str):
        path = Path(source['path']).expanduser()
        source['path'] = str((path if path.is_absolute() else Path(directory) / path).resolve())
    return asset


def expand_scene_assets(scene, directory):
    """Resolve asset refs relative to their declaration, then derive initial frames."""
    result = copy.deepcopy(scene)
    for obj in result.get('objects', {}).values():
        if 'asset' in obj:
            obj['asset'] = _asset_profile(obj['asset'], directory)
    validate_scene_objects(result)
    _, frames = resolve_scene_objects(result)
    result['frames'] = frames
    return result


def validate_scene_objects(scene):
    objects = scene.get('objects', {})
    if not isinstance(objects, dict):
        raise ValueError('scene.objects must be an object')
    for name, obj in objects.items():
        if not isinstance(name, str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError('scene object name must be an identifier')
        _fields(obj, {'label', 'roles', 'asset', 'placement', 'scale'}, f'scene.objects.{name}')
        if not isinstance(obj.get('roles', []), list) or not all(isinstance(r, str) and r for r in obj.get('roles', [])):
            raise ValueError('object roles must be strings')
        if 'asset' not in obj:
            if 'placement' in obj or 'scale' in obj:
                raise ValueError('physical placement requires an asset')
            continue
        validate_asset(obj['asset'])
        if any(v <= 0 for v in vector(obj.get('scale', [1, 1, 1]), 3, 'object.scale')):
            raise ValueError('object scale must be positive')
        placement = obj.get('placement', {})
        _fields(placement, {'frame', 'xy', 'position', 'quaternion_wxyz', 'grounding', 'clearance'}, 'object.placement')
        if not isinstance(placement.get('frame'), str):
            raise ValueError('object placement requires an explicit frame')
        pose_fields(placement, 'object.placement')
        if 'xy' in placement:
            vector(placement['xy'], 2, 'object.placement.xy')
            if 'position' in placement:
                raise ValueError('placement cannot specify both xy and position')
        if placement.get('grounding') not in (None, 'geometry_min'):
            raise ValueError('unsupported object grounding')
        clearance = placement.get('clearance', 0.)
        if type(clearance) not in (int, float) or not math.isfinite(clearance) or clearance < 0:
            raise ValueError('placement clearance must be finite and nonnegative')
        if 'xy' in placement and placement.get('grounding') != 'geometry_min':
            raise ValueError('xy placement requires geometry_min grounding')


def resolve_scene_objects(scene):
    """Return independent world poses; attachment names describe initial poses only."""
    objects, frames, resolved = scene.get('objects', {}), copy.deepcopy(scene.get('frames', {})), {}
    physical = {name: obj for name, obj in objects.items() if 'asset' in obj}
    def frame_pose(name, visiting):
        if name == 'world':
            return np.array([0., 0., 0., 1., 0., 0., 0.]), None
        if name in visiting:
            raise ValueError('cyclic physical scene placement')
        owner, dot, surface_name = name.partition('.')
        if owner in physical:
            item = resolve(owner, {*visiting, name})
            if not dot:
                return np.array(item['pose_world']), None
            surface = item['asset'].get('surfaces', {}).get(surface_name)
            if surface is None:
                raise ValueError('unknown asset support surface: ' + name)
            if item['asset']['physics']['body_type'] != 'static':
                raise ValueError('support placement requires a static support asset')
            scale = np.array(item['scale'])
            if not np.allclose(surface['quaternion_wxyz'], [1, 0, 0, 0]):
                if not np.allclose(scale, scale[0]):
                    raise ValueError('rotated support frames require uniform instance scale')
            local = [*(np.array(surface['position']) * scale), *surface['quaternion_wxyz']]
            bounds = {key: (np.array(surface['usable_bounds'][key]) * scale[:2]).tolist() for key in ('min', 'max')}
            return compose_pose(item['pose_world'], local), bounds
        if name == 'base' or name not in frames:
            raise ValueError('physical object requires a world-anchored frame: ' + name)
        config = frames[name]
        parent, _ = frame_pose(config['parent'], {*visiting, name})
        position, quaternion = pose_fields(config, 'scene frame')
        return compose_pose(parent, position + quaternion), None

    def resolve(name, visiting):
        if name in resolved:
            return resolved[name]
        obj = physical[name]
        placement = obj['placement']
        parent, bounds = frame_pose(placement['frame'], visiting)
        position, quaternion = pose_fields(placement, 'object placement')
        if 'xy' in placement:
            position[:2] = placement['xy']
        scale = obj.get('scale', [1., 1., 1.])
        vertices = np.concatenate([p['vertices'] for p in transformed_primitives(obj['asset'], scale)])
        rotated = vertices @ rotation_matrix(quaternion).T
        if placement.get('grounding') == 'geometry_min':
            if 'position' in placement and abs(position[2]) > 1e-12:
                raise ValueError('geometry_min computes height; use clearance rather than position.z')
            position[2] = -float(rotated[:, 2].min()) + placement.get('clearance', 0.)
        placed = rotated + position
        if bounds and (np.any(placed[:, :2].min(0) < np.array(bounds['min']) - 1e-8)
                       or np.any(placed[:, :2].max(0) > np.array(bounds['max']) + 1e-8)):
            raise ValueError(f'{name} geometry extends beyond usable support bounds')
        pose = compose_pose(parent, position + quaternion)
        world = vertices @ rotation_matrix(pose[3:]).T + pose[:3]
        item = {'name': name, 'asset': copy.deepcopy(obj['asset']), 'scale': list(scale),
                'pose_world': pose.tolist(), 'bounds_world': {'min': world.min(0).tolist(), 'max': world.max(0).tolist()},
                'initial_frame': placement['frame']}
        resolved[name] = item
        return item

    for name in physical:
        resolve(name, {name})
    for name, item in resolved.items():
        frames[name] = {'parent': 'world', 'position': item['pose_world'][:3], 'quaternion_wxyz': item['pose_world'][3:]}
        for key, surface in item['asset'].get('surfaces', {}).items():
            frames[name + '.' + key] = {'parent': name, 'position': (np.array(surface['position']) * item['scale']).tolist(),
                                      'quaternion_wxyz': surface['quaternion_wxyz']}
    return list(resolved.values()), frames
