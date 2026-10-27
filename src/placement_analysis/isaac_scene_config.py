"""CPU-only validation and configuration of the supported Isaac scene inputs."""
import copy
import math
import re

CAMERA_BODIES = {'head_rgbd': 'astribot_head_link_2',
                 'left_wrist_rgbd': 'astribot_arm_left_link_7',
                 'right_wrist_rgbd': 'astribot_arm_right_link_7'}
SUPPORTED_ROBOT_ROOT = '/World/astribot_whole_body'


def _fields(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f'{label}: unknown fields or non-object')


def vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length or not all(
            type(v) in (int, float) and math.isfinite(v) for v in value):
        raise ValueError(f'{label} requires {length} finite numbers')
    return list(value)


def pose_fields(value, label):
    position = vector(value.get('position', [0., 0., 0.]), 3, label + '.position')
    quaternion = vector(value.get('quaternion_wxyz', [1., 0., 0., 0.]), 4, label + '.quaternion_wxyz')
    if abs(sum(v*v for v in quaternion) - 1.) > 1e-6:
        raise ValueError(f'{label} quaternion must have unit norm')
    return position, quaternion


def normalize_scene_simulation(value=None):
    value = {} if value is None else copy.deepcopy(value)
    _fields(value, {'ground', 'light', 'cameras', 'background'}, 'scene.simulation')
    result = {'ground': {'enabled': True},
              'light': {'enabled': True, 'intensity': 3000., 'color': [.75, .75, .75]},
              'cameras': {}}
    for name in ('ground', 'light'):
        item = value.get(name, {})
        _fields(item, {'enabled', 'color'} | ({'intensity'} if name == 'light' else set()), f'scene.{name}')
        if type(item.get('enabled', True)) is not bool:
            raise ValueError(f'scene.{name}.enabled must be boolean')
        result[name].update(item)
        if 'color' in item:
            color = vector(item['color'], 3, f'scene.{name}.color')
            if any(v < 0 or v > 1 for v in color):
                raise ValueError('scene colors must be in [0, 1]')
            result[name]['color'] = color
        if name == 'light':
            intensity = result[name]['intensity']
            if type(intensity) not in (int, float) or not math.isfinite(intensity) or intensity < 0:
                raise ValueError('scene.light.intensity must be finite and nonnegative')
    cameras = value.get('cameras', {})
    _fields(cameras, set(CAMERA_BODIES), 'scene.cameras')
    for name, camera in cameras.items():
        _fields(camera, {'enabled', 'width', 'height', 'body'}, f'camera {name}')
        if type(camera.get('enabled', True)) is not bool:
            raise ValueError('camera.enabled must be boolean')
        for dimension in ('width', 'height'):
            if dimension in camera and (type(camera[dimension]) is not int or camera[dimension] <= 0):
                raise ValueError('camera dimensions must be positive integers')
        body = camera.get('body', CAMERA_BODIES[name])
        if not isinstance(body, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', body):
            raise ValueError('camera.body must be a single USD body identifier')
        result['cameras'][name] = {**camera, 'body': body, 'enabled': camera.get('enabled', True)}
    if 'background' in value:
        background = value['background']
        _fields(background, {'usd_path', 'position', 'quaternion_wxyz'}, 'scene.background')
        if not isinstance(background.get('usd_path'), str) or not background['usd_path']:
            raise ValueError('background.usd_path must be a nonempty path')
        position, quaternion = pose_fields(background, 'scene.background')
        result['background'] = {**background, 'position': position, 'quaternion_wxyz': quaternion}
    return result


def spawn_scene_environment(sim_utils, settings):
    """Called only after AppLauncher; injectable config classes permit CPU tests."""
    scene = normalize_scene_simulation(settings)
    ground = scene['ground']
    if ground['enabled']:
        config = sim_utils.GroundPlaneCfg(**({'color': tuple(ground['color'])} if 'color' in ground else {}))
        config.func('/World/ground', config)
    light = scene['light']
    if light['enabled']:
        config = sim_utils.DomeLightCfg(intensity=light['intensity'], color=tuple(light['color']))
        config.func('/World/light', config)
    if 'background' in scene:
        background = scene['background']
        config = sim_utils.UsdFileCfg(usd_path=background['usd_path'])
        config.func('/World/scene_background', config, translation=tuple(background['position']),
                    orientation=tuple(background['quaternion_wxyz']))


def camera_prim_path(name, body, robot_root=SUPPORTED_ROBOT_ROOT):
    if robot_root != SUPPORTED_ROBOT_ROOT:
        raise ValueError('Isaac scene adapter currently requires /World/astribot_whole_body robot root')
    if name not in CAMERA_BODIES or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', body):
        raise ValueError('unsupported camera name or body path')
    return f'/World/envs/env_0/Robot/astribot_whole_body/{body}/{name}'
