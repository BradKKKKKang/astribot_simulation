"""Pure CPU configuration, fingerprint and robot-mapping contracts."""
import copy
import json
from pathlib import Path

import pytest

from src.simulation_config.experiment_config import resolve_experiment, read_json
from src.simulation_config.robot_profile import default_robot_profile, RobotProfile
from src.placement_analysis.task_config import DEFAULT_DETECTOR_CONFIG, load_task_config


def expanded():
    result = resolve_experiment(DEFAULT_DETECTOR_CONFIG)
    return {'schema_version': 2, 'kind': 'placement_test', **copy.deepcopy(result['sections'])}


def resolve(tmp_path, value):
    path = tmp_path / 'experiment.json'
    path.write_text(json.dumps(value))
    return resolve_experiment(path)


def test_checked_in_modular_configs_preserve_behavior_and_pending():
    press = load_task_config(DEFAULT_DETECTOR_CONFIG, validate_identification=True)
    assert press['type'] == 'press' and press['press_min_depth_m'] == .02
    assert press['tcp']['left']['position'] == [0., -.195, 0.]
    pen = load_task_config(DEFAULT_DETECTOR_CONFIG.with_name('penholder.json'), validate_identification=True)
    assert pen['type'] == 'gripper' and pen['grasp_onset_threshold'] == 5
    assert pen['identification']['object'] == '笔'
    pending = load_task_config(DEFAULT_DETECTOR_CONFIG.with_name('cap_pen.json'), require_ready=False)
    assert pending['type'] is None and pending['event_rules'] is None
    with pytest.raises(ValueError, match='needs_annotation'):
        load_task_config(DEFAULT_DETECTOR_CONFIG.with_name('cap_pen.json'))


def test_legacy_roundtrip_without_jev_request_is_extractable(tmp_path):
    legacy = load_task_config(DEFAULT_DETECTOR_CONFIG)
    legacy.pop('_resolved_sections'); legacy.pop('_fingerprints')
    legacy['identification'] = {'enabled': True}
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(legacy))
    result = load_task_config(path)
    assert result['type'] == 'press'
    with pytest.raises(ValueError, match='identification.camera'):
        load_task_config(path, validate_identification=True)


def test_duplicate_unknown_fields_and_invalid_schema_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match='duplicate'):
        read_json(path)
    value = expanded(); value['candidates']['press_min_dept_m'] = .1
    with pytest.raises(ValueError, match='unknown'):
        resolve(tmp_path, value)
    value = expanded(); value['schema_version'] = True
    with pytest.raises(ValueError, match='schema_version'):
        resolve(tmp_path, value)
    value = expanded(); value['robot']['tcp']['left']['quaternion_wxyz'] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match='unit norm'):
        resolve(tmp_path, value)


def test_reference_overrides_relative_to_referring_file_and_cycles(tmp_path):
    value = expanded()
    profiles = tmp_path / 'profiles'; profiles.mkdir()
    (profiles / 'base.json').write_text(json.dumps(value['candidates']))
    (profiles / 'derived.json').write_text(json.dumps({'ref': 'base.json', 'overrides': {'press_min_depth_m': .03}}))
    value['candidates'] = {'ref': 'profiles/derived.json', 'overrides': {'press_min_retract_m': .02}}
    result = resolve(tmp_path, value)
    assert result['settings']['press_min_depth_m'] == .03
    assert result['settings']['press_min_retract_m'] == .02
    (profiles / 'base.json').write_text(json.dumps({'ref': 'derived.json'}))
    with pytest.raises(ValueError, match='cyclic'):
        resolve(tmp_path, value)


def test_stage_fingerprints_do_not_overinvalidate(tmp_path):
    value = expanded()
    first = resolve(tmp_path, value)['fingerprints']
    value['jev']['transport']['timeout_s'] = 93
    timeout = resolve(tmp_path, value)['fingerprints']
    assert timeout['jev_transport'] != first['jev_transport']
    for key in ('extraction', 'jev_request', 'jev_decision', 'robot_control'):
        assert timeout[key] == first[key]
    value['jev']['decision']['min_probability'] = .7
    decision = resolve(tmp_path, value)['fingerprints']
    assert decision['jev_decision'] != timeout['jev_decision']
    assert decision['jev_request'] == timeout['jev_request']
    value['robot']['tcp']['left']['position'][0] = .01
    tcp = resolve(tmp_path, value)['fingerprints']
    assert tcp['robot_kinematics'] != decision['robot_kinematics']
    assert tcp['robot_control'] == decision['robot_control']
    assert tcp['jev_request'] == decision['jev_request']
    value['jev']['transport']['endpoint'] = 'https://example.invalid/inference'
    assert resolve(tmp_path, value)['fingerprints']['jev_request'] != tcp['jev_request']


def test_choice_order_is_semantic_but_json_object_order_is_not(tmp_path):
    value = expanded(); before = resolve(tmp_path, value)['fingerprints']
    reordered = dict(reversed(list(value.items())))
    assert resolve(tmp_path, reordered)['fingerprints'] == before
    value['jev']['request']['choice_order'].reverse()
    assert resolve(tmp_path, value)['fingerprints']['jev_request'] != before['jev_request']


def test_robot_mapping_matches_existing_physical_contracts():
    profile = default_robot_profile()
    assert len(profile.dataset_joint_names) == 25 and len(profile.joint_names) == 22
    assert profile.sdk_parts[2] == ('astribot_gripper_left', 14, 15)
    values = [0.] * 22
    values[profile.joint_names.index('astribot_gripper_left_joint_1')] = 50
    values[profile.joint_names.index('astribot_gripper_right_joint_1')] = 101
    joints, grippers = profile.feedback_joint_positions(profile.joint_names, values)
    assert grippers == [50, 100]
    assert joints['astribot_gripper_left_joint_L1'] == -.465
    assert joints['astribot_gripper_right_joint_R1'] == .93
    assert len(joints) == 32
    with pytest.raises(ValueError, match='every configured joint'):
        profile.feedback_joint_positions(profile.joint_names[:-1], values[:-1])
    invalid = profile.as_dict(); invalid['parts'][1]['command_indices'][0] = 3
    with pytest.raises(ValueError):
        RobotProfile(invalid)


def test_custom_robot_mapping_not_hardcoded():
    data = default_robot_profile().as_dict()
    data['parts'][0]['model_joints'][0] = 'custom_torso_joint'
    data['grippers']['left']['radians_per_percent'] = .01
    data['grippers']['left']['model_joints'] = {'custom_finger': -1}
    profile = RobotProfile(data)
    joints, _ = profile.feedback_joint_positions(profile.joint_names, [50.] * 22)
    assert joints['custom_torso_joint'] == 50
    assert joints['custom_finger'] == -.5


@pytest.mark.parametrize('change', [
    lambda v: v['runtime'].update(alignment={'max_gap_s': 0}),
    lambda v: v.update(scene={}),
    lambda v: v['jev']['decision'].update(min_probability=1.1),
    lambda v: v['jev']['transport'].update(api_key='never-print-this'),
])
def test_invalid_nested_configuration(tmp_path, change):
    value = expanded(); change(value)
    with pytest.raises(ValueError):
        resolve(tmp_path, value)


def test_choice_order_survives_canonical_serialization(tmp_path):
    value = expanded()
    order = value['jev']['request']['choice_order']
    path = tmp_path / 'sorted.json'
    path.write_text(json.dumps(value, sort_keys=True))
    result = resolve_experiment(path)
    assert result['settings']['identification']['choice_order'] == order
    assert result['fingerprints']['jev_request'] == resolve(tmp_path, value)['fingerprints']['jev_request']
    value['jev']['request']['choice_order'] = ['red', 'red', 'blue']
    with pytest.raises(ValueError, match='choice_order'):
        resolve(tmp_path, value)


def test_simulation_profile_builds_cpu_config_and_control_fingerprints(tmp_path):
    from src.simulation_config.robot_profile import build_simulation_config
    value = expanded()
    before = resolve(tmp_path, value)['fingerprints']
    value['robot']['simulation_overrides'] = {'gripper_left_control': {'stiffness': 900.}}
    after = resolve(tmp_path, value)['fingerprints']
    assert after['robot_control'] != before['robot_control']
    assert after['robot_kinematics'] == before['robot_kinematics']
    config = build_simulation_config(value['robot'])
    assert config['gripper_left_control']['stiffness'] == 900
    profile = RobotProfile(value['robot'])
    assert config['robot_list'] == [name for name, _ in profile.part_configs]
    assert config['joint_names_list'][2] == ['astribot_gripper_left_joint_L1']
    assert config['vel_compensation_list'] == [True, True, False, True, False, True]
    value['robot']['simulation_overrides']['gripper_left_control']['stiffnes'] = 20
    with pytest.raises(ValueError, match='unknown simulation override'):
        build_simulation_config(value['robot'])


def test_robot_mapping_rejects_duplicate_fixed_indices_and_gripper_owners():
    value = default_robot_profile().as_dict()
    value['fixed_chassis']['command_indices'].append(0)
    with pytest.raises(ValueError, match='cover dataset exactly'):
        RobotProfile(value)
    value = default_robot_profile().as_dict()
    value['grippers']['right']['part'] = value['grippers']['left']['part']
    value['grippers']['right']['feedback_joint'] = value['grippers']['left']['feedback_joint']
    with pytest.raises(ValueError, match='unique'):
        RobotProfile(value)


@pytest.mark.parametrize('change', [
    lambda value: value.pop('name'),
    lambda value: value['robot']['parts'][0].update(command_indices=3),
    lambda value: value['robot']['parts'][0].update(feedback_joints=None),
])
def test_malformed_schema_raises_value_error(tmp_path, change):
    value = expanded(); change(value)
    with pytest.raises(ValueError):
        resolve(tmp_path, value)


def test_default_robot_profile_rejects_duplicate_json_keys(tmp_path, monkeypatch):
    from src.simulation_config import robot_profile
    path = tmp_path / 'robot.json'
    path.write_text('{"name":"one","name":"two"}')
    monkeypatch.setattr(robot_profile, 'DEFAULT_ROBOT_PROFILE', path)
    with pytest.raises(ValueError, match='duplicate robot profile'):
        robot_profile.default_robot_profile()


def test_resolver_preflight_rejects_unknown_simulator_overrides(tmp_path):
    value = expanded()
    value['robot']['simulation_overrides'] = {'gripper_left_control': {'stiffnes': 900}}
    with pytest.raises(ValueError, match='unknown simulation override'):
        resolve(tmp_path, value)
    value = expanded()
    value['robot'].pop('simulation_config')
    value['robot']['simulation_overrides'] = {}
    # External capture profiles may omit simulator configuration entirely.
    assert resolve(tmp_path, value)['settings']['type'] == 'press'


def test_v2_placement_has_no_evaluation_dependency(tmp_path, monkeypatch):
    import builtins
    from src.simulation_config.scene_profile import build_experiment_simulation_config
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.startswith(('src.placement_analysis', 'src.evaluation', 'rclpy', 'isaaclab', 'pxr')):
            raise AssertionError('unexpected workflow/runtime import: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    resolved = resolve_experiment(DEFAULT_DETECTOR_CONFIG)
    assert resolved['kind'] == 'placement_test'
    assert not {'scene', 'task', 'success'} & resolved['sections'].keys()
    assert resolved['settings']['event_rules'] == {'exclude_initial': True}
    config = build_experiment_simulation_config(resolved)
    assert config['scene_simulation']['ground']['enabled'] is True
    assert config['robot_list'] == [p['name'] for p in resolved['sections']['robot']['parts']]


def test_v1_v2_preserve_replay_extraction_and_request_identity(tmp_path):
    new = resolve_experiment(DEFAULT_DETECTOR_CONFIG)
    sections = new['sections']
    legacy = {key: copy.deepcopy(sections[key]) for key in ('robot', 'candidates', 'jev', 'source', 'runtime', 'output')}
    rules = legacy['candidates'].pop('selection')
    legacy['source'].pop('dataset')
    legacy.update(schema_version=1, scene={'objects': {}, 'regions': {}},
                  task={**new['settings']['task'], 'event_rules': rules}, success={'predicates': []})
    old = resolve(tmp_path, legacy)
    for key in ('robot_control', 'robot_kinematics', 'robot_tcp', 'scene_simulation',
                'extraction', 'jev_request', 'jev_decision', 'jev_transport'):
        assert old['fingerprints'][key] == new['fingerprints'][key], key
    strip = lambda value: {k: v for k, v in value.items() if not k.startswith('_')}
    assert strip(old['settings']) == strip(new['settings'])


def test_evaluation_and_placement_reject_each_others_fields(tmp_path):
    for field in ('scene', 'task', 'success'):
        value = expanded(); value[field] = {}
        with pytest.raises(ValueError, match='unknown fields'):
            resolve(tmp_path, value)
    evaluation = DEFAULT_DETECTOR_CONFIG.parent.parent / 'evaluation/press_cube.json'
    resolved = resolve_experiment(evaluation)
    assert 'candidates' not in resolved['sections'] and 'settings' not in resolved
    with pytest.raises(ValueError, match='placement_test'):
        load_task_config(evaluation)
    value = {'schema_version': 2, 'kind': 'evaluation', **resolved['sections'], 'jev': {}}
    with pytest.raises(ValueError, match='unknown fields'):
        resolve(tmp_path, value)


def test_source_root_is_optional_until_source_resolution(tmp_path):
    value = expanded()
    value['source'].pop('dataset_root')
    resolved = resolve(tmp_path, value)
    assert resolved['settings']['dataset_root'] is None
    from scripts import replay_extract_event as replay
    path = tmp_path / 'experiment.json'
    assert replay.main(['--experiment', str(path), '--dry-run']) == 1
