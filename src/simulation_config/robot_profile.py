"""Validated CPU-only robot mapping shared by replay, feedback and extraction."""
import copy
import json
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_PROFILE = ROOT / 'config/robots/astribot_s1.json'


def _keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f'{label}: expected object with fields {sorted(allowed)}')


def _finite_vector(value, size, label):
    if not isinstance(value, list) or len(value) != size or not all(
            type(v) in (int, float) and math.isfinite(v) for v in value):
        raise ValueError(f'{label}: expected {size} finite numbers')


class RobotProfile:
    """Names and explicit command-to-articulation mappings; no SDK imports."""

    def __init__(self, value):
        _keys(value, {'name', 'model_path', 'root_path', 'base_body', 'tcp', 'tcp_calibration',
                      'parts', 'dataset_joint_names', 'fixed_chassis', 'grippers',
                      'simulation_config', 'simulation_overrides', 'initial_joint_positions'}, 'robot')
        self._data = copy.deepcopy(value)
        initial = value.get('initial_joint_positions', {})
        if not isinstance(initial, dict) or not all(isinstance(k, str) and k and
                type(v) in (int, float) and math.isfinite(v) for k, v in initial.items()):
            raise ValueError('robot.initial_joint_positions must map model joints to finite radians')
        for field in ('name', 'model_path', 'root_path', 'base_body'):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ValueError(f'robot.{field} is required')
        if not value['root_path'].startswith('/'):
            raise ValueError('robot.root_path must be an absolute USD prim path')
        tcp = value.get('tcp')
        if not isinstance(tcp, dict) or set(tcp) != {'left', 'right'}:
            raise ValueError('robot.tcp requires left and right endpoints')
        for side, endpoint in tcp.items():
            _keys(endpoint, {'body', 'position', 'quaternion_wxyz'}, f'robot.tcp.{side}')
            if not isinstance(endpoint.get('body'), str) or not endpoint['body']:
                raise ValueError('TCP body is required')
            _finite_vector(endpoint.get('position'), 3, 'TCP position')
            _finite_vector(endpoint.get('quaternion_wxyz'), 4, 'TCP quaternion')
            if abs(sum(v*v for v in endpoint['quaternion_wxyz']) - 1) > 1e-6:
                raise ValueError('TCP quaternion must have unit norm')
        if 'simulation_config' in value and (not isinstance(value['simulation_config'], str) or not value['simulation_config']):
            raise ValueError('robot.simulation_config must be a nonempty repo-root path')
        if not isinstance(value.get('simulation_overrides', {}), dict):
            raise ValueError('robot.simulation_overrides must be an object')
        parts = value.get('parts')
        if not isinstance(parts, list) or not parts:
            raise ValueError('robot.parts is required')
        names, feedback, articulation, indices = [], [], [], []
        for part in parts:
            _keys(part, {'name', 'command_indices', 'feedback_joints', 'model_joints'}, 'robot part')
            name = part.get('name')
            if not isinstance(name, str) or not name:
                raise ValueError('robot part name is required')
            names.append(name)
            command = part.get('command_indices', [])
            joints, model = part.get('feedback_joints', []), part.get('model_joints', [])
            if (not isinstance(command, list) or not isinstance(joints, list) or not isinstance(model, list)
                    or not command or any(type(i) is not int or i < 0 for i in command)
                    or command != list(range(command[0], command[0] + len(command)))
                    or len(command) != len(joints) or len(model) != len(joints)
                    or not all(isinstance(s, str) and s for s in joints + model)):
                raise ValueError('robot part requires contiguous command indices and equally sized joint lists')
            indices.extend(command); feedback.extend(joints); articulation.extend(model)
        if any(len(items) != len(set(items)) for items in (names, feedback, articulation, indices)):
            raise ValueError('robot part names, joints and command indices must be unique')
        dataset = value.get('dataset_joint_names')
        if not isinstance(dataset, list) or not dataset or not all(isinstance(s, str) and s for s in dataset) or len(dataset) != len(set(dataset)):
            raise ValueError('robot.dataset_joint_names must be unique names')
        fixed = value.get('fixed_chassis')
        _keys(fixed, {'part', 'command_indices'}, 'robot.fixed_chassis')
        fixed_indices = fixed.get('command_indices', [])
        if (not isinstance(fixed.get('part'), str) or not fixed['part'] or not isinstance(fixed_indices, list)
                or any(type(i) is not int or i < 0 for i in fixed_indices)
                or len(indices + fixed_indices) != len(dataset)
                or len(set(indices + fixed_indices)) != len(dataset)
                or set(indices + fixed_indices) != set(range(len(dataset)))):
            raise ValueError('robot command indices must cover dataset exactly, including fixed chassis')
        grippers = value.get('grippers')
        if not isinstance(grippers, dict) or set(grippers) != {'left', 'right'}:
            raise ValueError('robot.grippers requires left and right')
        expanded_names = []
        for side, gripper in grippers.items():
            _keys(gripper, {'part', 'feedback_joint', 'radians_per_percent', 'model_joints', 'measured_joint'}, f'gripper {side}')
            if gripper.get('part') not in names or gripper.get('feedback_joint') not in feedback:
                raise ValueError('gripper must reference a configured part and feedback joint')
            scale = gripper.get('radians_per_percent')
            mapping = gripper.get('model_joints')
            if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
                raise ValueError('gripper.radians_per_percent must be positive')
            if not isinstance(mapping, dict) or not mapping or not all(
                    isinstance(k, str) and k and type(v) in (int, float) and math.isfinite(v) and v != 0
                    for k, v in mapping.items()):
                raise ValueError('gripper.model_joints must map names to finite nonzero directions')
            if gripper.get('measured_joint', next(iter(mapping))) not in mapping:
                raise ValueError('gripper.measured_joint must be a configured model joint')
            owner = next(part for part in parts if part['name'] == gripper['part'])
            if owner['feedback_joints'] != [gripper['feedback_joint']]:
                raise ValueError('gripper must own exactly its configured feedback joint')
            expanded_names.extend(mapping)
        ordinary = {model for part in parts if part['name'] not in {g['part'] for g in grippers.values()}
                    for model in part['model_joints']}
        if (len(expanded_names) != len(set(expanded_names)) or ordinary.intersection(expanded_names)
                or len({g['part'] for g in grippers.values()}) != len(grippers)):
            raise ValueError('gripper parts and model joints must be unique and separate from other model joints')
        if value.get('simulation_overrides'):
            build_simulation_config(self)

    @classmethod
    def from_dict(cls, value):
        return cls(value)

    def as_dict(self):
        return copy.deepcopy(self._data)

    @property
    def part_configs(self):
        return tuple((p['name'], len(p['feedback_joints'])) for p in self._data['parts'])

    @property
    def joint_names(self):
        return tuple(j for p in self._data['parts'] for j in p['feedback_joints'])

    @property
    def feedback_parts(self):
        return {joint: part['name'] for part in self._data['parts'] for joint in part['feedback_joints']}

    @property
    def anchor_joint(self):
        return self.joint_names[0]

    @property
    def sdk_parts(self):
        return tuple((p['name'], p['command_indices'][0], p['command_indices'][-1]+1) for p in self._data['parts'])

    @property
    def dataset_joint_names(self):
        return tuple(self._data['dataset_joint_names'])

    @property
    def command_indices(self):
        return tuple(i for part in self._data['parts'] for i in part['command_indices'])

    @property
    def chassis_indices(self):
        return tuple(self._data['fixed_chassis']['command_indices'])

    @property
    def chassis_part(self):
        return self._data['fixed_chassis']['part']

    @property
    def gripper_command_indices(self):
        parts = {part['name']: part for part in self._data['parts']}
        return tuple(parts[self._data['grippers'][side]['part']]['command_indices'][0] for side in ('left', 'right'))

    @property
    def feedback_aliases(self):
        return {config.get('measured_joint', next(iter(config['model_joints']))): config['feedback_joint']
                for config in self._data['grippers'].values()}

    def control_provenance(self):
        result = {key: copy.deepcopy(self._data[key]) for key in
                  ('parts', 'dataset_joint_names', 'fixed_chassis', 'grippers')}
        result['simulation_overrides'] = copy.deepcopy(self._data.get('simulation_overrides', {}))
        if 'simulation_config' in self._data:
            path = Path(self._data['simulation_config']).expanduser()
            path = path if path.is_absolute() else ROOT / path
            result['simulation_config'] = self._data['simulation_config']
            result['simulation_config_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['model_path'] = self._data['model_path']
        model = self.model_path
        result['model_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else None
        result['root_path'] = self._data['root_path']
        if 'initial_joint_positions' in self._data:
            result['initial_joint_positions'] = copy.deepcopy(self._data['initial_joint_positions'])
        result['identity_basis'] = 'declared profile and config; not verified running simulator identity'
        return result

    @property
    def model_path(self):
        path = Path(self._data['model_path']).expanduser()
        return path if path.is_absolute() else ROOT / path

    @property
    def root_path(self):
        return self._data['root_path']

    @property
    def base_body(self):
        return self._data['base_body']

    @property
    def tcp(self):
        return copy.deepcopy(self._data['tcp'])

    def sdk_joint_positions(self, settings):
        joints = {}
        parts = {p['name']: p for p in self._data['parts']}
        for name, values in settings['joint_targets_rad'].items():
            if name not in parts or len(values) != len(parts[name]['model_joints']):
                raise ValueError(f'unknown or invalid SDK part target: {name}')
            joints.update(zip(parts[name]['model_joints'], map(float, values)))
        for side, value in settings['gripper_percent'].items():
            if side not in self._data['grippers'] or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError('invalid reference gripper target')
            config = self._data['grippers'][side]
            joints.update({name: value * sign * config['radians_per_percent']
                           for name, sign in config['model_joints'].items()})
        if not all(math.isfinite(v) for v in joints.values()):
            raise ValueError('reference joint targets must be finite')
        return joints

    def feedback_joint_positions(self, joint_names, values):
        if (len(joint_names) != len(values) or len(set(joint_names)) != len(joint_names)
                or set(joint_names) != set(self.joint_names) or not all(math.isfinite(v) for v in values)):
            raise ValueError('feedback must contain every configured joint exactly once with finite values')
        feedback = dict(zip(joint_names, values))
        grippers = {side: min(100., max(0., float(feedback[c['feedback_joint']])))
                    for side, c in self._data['grippers'].items()}
        gripper_joints = {c['feedback_joint'] for c in self._data['grippers'].values()}
        joints = {model: float(feedback[joint]) for part in self._data['parts']
                  for joint, model in zip(part['feedback_joints'], part['model_joints']) if joint not in gripper_joints}
        joints.update(self.sdk_joint_positions({'joint_targets_rad': {}, 'gripper_percent': grippers}))
        return joints, [grippers['left'], grippers['right']]


def default_robot_profile():
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate robot profile field: {key}')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f'nonfinite robot profile value: {value}')
    return RobotProfile(json.loads(DEFAULT_ROBOT_PROFILE.read_text(),
                                   object_pairs_hook=unique_pairs, parse_constant=invalid_constant))


def profile_for_settings(settings):
    sections = settings.get('_resolved_sections', {})
    if 'robot' in sections:
        return RobotProfile(sections['robot'])
    value = default_robot_profile().as_dict()
    for field in ('base_body', 'tcp', 'tcp_calibration'):
        if field in settings:
            value[field] = copy.deepcopy(settings[field])
    return RobotProfile(value)


def build_simulation_config(profile):
    """Build a declared simulation config on CPU; never launch or contact ROS.

    Config paths and model paths are relative to the repository root. Overrides
    may change existing YAML fields only; robot/part mappings come from profile.
    The returned absolute model path is ready for the environment factory.
    """
    import yaml
    profile = profile if isinstance(profile, RobotProfile) else RobotProfile(profile)
    data = profile.as_dict()
    if 'simulation_config' not in data:
        raise ValueError('robot profile has no simulation_config')
    path = Path(data['simulation_config']).expanduser()
    path = path if path.is_absolute() else ROOT / path
    class StrictLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f'duplicate simulation configuration field: {key}')
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    config = yaml.load(path.read_text(), Loader=StrictLoader)
    if not isinstance(config, dict):
        raise ValueError('simulation configuration must be an object')
    def merge(base, overrides):
        for key, value in overrides.items():
            if key not in base:
                raise ValueError(f'unknown simulation override field: {key}')
            if isinstance(value, dict) and isinstance(base[key], dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
    overrides = data.get('simulation_overrides', {})
    if set(overrides) & {'model_path', 'robot_list', 'joint_names_list'}:
        raise ValueError('model and joint mappings are owned by the robot profile, not simulation overrides')
    merge(config, overrides)
    # Retain configured per-part options when changing SDK replay ordering.
    prior_parts = config.get('robot_list', [])
    compensation = dict(zip(prior_parts, config.get('vel_compensation_list', [])))
    if any(part['name'] not in compensation for part in data['parts']):
        raise ValueError('simulation config must declare velocity compensation for every robot part')
    grippers = {item['part']: item for item in data['grippers'].values()}
    config['model_path'] = str(profile.model_path)
    config['robot_list'] = [part['name'] for part in data['parts']]
    config['joint_names_list'] = [([grippers[part['name']].get('measured_joint', next(iter(grippers[part['name']]['model_joints'])))]
                                  if part['name'] in grippers else part['model_joints']) for part in data['parts']]
    config['vel_compensation_list'] = [compensation[name] for name in config['robot_list']]
    # Reject invalid override numerics before a caller can construct a simulator.
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('simulation configuration numbers must be finite')
        if isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
    finite(config)
    return config
