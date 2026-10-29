"""Scene configuration shared by simulator composition and offline coordinates."""
import copy
import hashlib
import math
from pathlib import Path

from src.simulation_config.isaac_scene_config import (
    CAMERA_BODIES, SUPPORTED_ROBOT_ROOT, normalize_scene_simulation, pose_fields,
)
from .robot_profile import ROOT, RobotProfile, build_simulation_config


def validate_scene_frames(scene):
    frames = scene.get('frames', {})
    if not isinstance(frames, dict):
        raise ValueError('scene.frames must be a named object')
    for name, frame in frames.items():
        if not isinstance(name, str) or not name or name in ('world', 'base') or not isinstance(frame, dict) or set(frame) - {
                'parent', 'position', 'quaternion_wxyz'}:
            raise ValueError('scene frame requires a unique name and parent/pose fields')
        if not isinstance(frame.get('parent'), str) or frame['parent'] not in {'world', 'base', *frames}:
            raise ValueError('scene frame references unknown parent')
        pose_fields(frame, f'scene.frames.{name}')
    def check(name, visiting):
        if name in ('world', 'base'):
            return
        if name in visiting:
            raise ValueError('cyclic scene frame calibration')
        check(frames[name]['parent'], {*visiting, name})
    for name in frames:
        check(name, set())
    normalize_scene_simulation(scene.get('simulation'))


def frame_coordinates(event, scene, frame):
    """Transform a recorded TCP into a calibrated frame; return None if missing."""
    from .poses import compose_pose, relative_pose
    frames = scene.get('frames', {})
    def resolved(name):
        if name in ('world', 'base'):
            return name, [0., 0., 0., 1., 0., 0., 0.]
        config = frames[name]
        anchor, parent = resolved(config['parent'])
        position, quaternion = pose_fields(config, f'scene.frames.{name}')
        return anchor, compose_pose(parent, position + quaternion)
    anchor, pose = resolved(frame)
    tcp = event.get('tcp_pose_' + anchor)
    if (not isinstance(tcp, (list, tuple)) or len(tcp) != 7
            or not all(type(value) in (int, float) and math.isfinite(value) for value in tcp)):
        return None
    return relative_pose(pose, tcp)[:3].tolist()


def scene_simulation_provenance(scene):
    """Only physical scene settings affect new capture identity, not task regions."""
    result = normalize_scene_simulation(scene.get('simulation'))
    if 'background' in result:
        path = Path(result['background']['usd_path']).expanduser()
        path = path if path.is_absolute() else ROOT / path
        result['background']['usd_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if any('asset' in obj for obj in scene.get('objects', {}).values()):
        from .scene_assets import resolve_scene_objects
        result['objects'], _ = resolve_scene_objects(scene)
    return result


def build_experiment_simulation_config(resolved):
    """Compose an Isaac config without importing the factory, ROS or Isaac Lab."""
    sections = resolved['sections']
    profile = RobotProfile(sections['robot'])
    if 'initial_pose' in sections.get('task', {}):
        from .initial_pose import initial_joint_positions
        if 'robot_placement' not in sections:
            raise ValueError('task.initial_pose requires robot_placement for initialization and reset')
        robot = profile.as_dict()
        robot['initial_joint_positions'] = {
            **robot.get('initial_joint_positions', {}),
            **initial_joint_positions(sections['task']['initial_pose'], profile),
        }
        profile = RobotProfile(robot)
    config = build_simulation_config(profile)
    if config.get('simulator_type') != 'IsaacLab':
        raise ValueError('experiment scene composition currently supports IsaacLab only')
    if profile.root_path != SUPPORTED_ROBOT_ROOT:
        raise ValueError('Isaac experiment requires /World/astribot_whole_body robot root')
    scene = sections.get('scene', {})
    validate_scene_frames(scene)
    simulation = normalize_scene_simulation(scene.get('simulation'))
    if 'background' in simulation:
        background = Path(simulation['background']['usd_path']).expanduser()
        background = background if background.is_absolute() else ROOT / background
        background = background.resolve(strict=True)
        if not background.is_file() or background.suffix.lower() not in ('.usd', '.usda', '.usdc'):
            raise ValueError('scene background must reference a USD file')
        simulation['background']['usd_path'] = str(background)
    names = list(config.get('camera_names', []))
    if set(names) - set(CAMERA_BODIES):
        raise ValueError('Isaac experiment has unsupported configured camera names')
    resolutions = copy.deepcopy(config.get('camera_resolutions', {}))
    bodies = dict(CAMERA_BODIES)
    for name, values in simulation['cameras'].items():
        if values['enabled'] and name not in names:
            names.append(name)
        elif not values['enabled'] and name in names:
            names.remove(name)
        resolutions.setdefault(name, {}).update({key: values[key] for key in ('width', 'height') if key in values})
        bodies[name] = values['body']
    config.update(camera_names=names, camera_resolutions=resolutions, camera_bodies=bodies,
                  robot_root_path=profile.root_path, scene_simulation=simulation)
    from .scene_assets import resolve_scene_objects, validate_scene_objects
    validate_scene_objects(scene)
    objects, frames = resolve_scene_objects(scene)
    config['scene_objects'] = objects
    if sections.get('task', {}).get('reset') is not None:
        config['task_reset'] = copy.deepcopy(sections['task']['reset'])
        config['task_scene'] = copy.deepcopy(scene)
    config['robot_profile'] = profile.as_dict()
    if 'robot_placement' in sections:
        config['robot_placement'] = copy.deepcopy(sections['robot_placement'])
    config.update(copy.deepcopy(sections.get('runtime', {}).get('simulation', {})))
    config['experiment_provenance'] = {'robot_control': profile.control_provenance(),
        'scene_simulation': scene_simulation_provenance(scene),
        'scene_frames': frames,
        'identity_basis': 'resolved launch inputs; not a runtime state attestation'}
    return config
