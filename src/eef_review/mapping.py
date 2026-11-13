"""Describe published/sample contracts without changing their source hashes."""
from .common import digest

PUBLISHED_DATASETS = {
    'ABC-130K', 'AgiBotDigitalWorld', 'AgiBotWorld2026', 'AgiBotWorldChallenge-2025',
    'Dexora_Real-World_Dataset', 'Galaxea-Open-World-Dataset-fix-coordinate-no-motion_filtered',
    'GenieSim3.0', 'HABIT', 'agibot_data', 'agibot_data_extracted_selected_x2', 'baihu-dataset-en_aligned',
}
UMI_DATASETS = {'FastUMI_aligned', 'HY_aligned', 'HiFi', 'realomin_10kh'}


def describe(group):
    contract, details, dataset = group['contract'], group['details'], group['dataset']
    if 'published_configuration' in contract and dataset in PUBLISHED_DATASETS:
        published = contract['published_configuration']
        action = published.get('action_kind')
        absolute = action in {
            'original matched commanded pose; no temporal shift or substitution',
            'absolute next-state target; final row holds state[t]',
            'EEF pose action[t] = transformed state arm pose[t+1], within episode',
            'state[t+1] within episode; final action=state[t]',
            'same-row published source absolute target; original arm_pose/end_pose TCP relation unverified',
        }
        return {'kind': 'published', 'profile': details.get('profile', 'published_' + dataset),
                'coordinates': published.get('coordinates', {}),
                'action_kind': 'absolute_pose' if absolute else action,
                'has_pose': details.get('eefAvailability', {}).get('hasPose'),
                'source': contract.get('source', ''), 'robot': contract.get('robot', ''),
                'configuration': contract.get('published_processing_id', ''),
                'pose_encoding': published.get('pose_encoding')}
    if contract.get('schema') == 'umi_aligned_sample_mapping_v1' and dataset in UMI_DATASETS:
        first_state = dataset == 'realomin_10kh'
        policies = contract.get('source_policies', {})
        absolute = (policies.get('action_layout', {}).get('type') == 'absolute_next_state_target'
                    or policies.get('umi', {}).get('action_temporal_alignment') == 'next_state_exact_within_episode; terminal row UNKNOWN')
        # HY's augmented EEF action is episode-local next-State pose with native
        # gripper action. Verified in extras/action_augmentation/semantics.json
        # and its completion receipt across all 22 registered source roots.
        if (dataset == 'HY_aligned'
                and policies.get('padding', {}).get('version') == 'praxis.hy-lumo-dimensions-inplace.v1'
                and policies.get('extras/pose_transform.json', {}).get('schema_version') == 'praxis.hy-rxminus90-rot6d.v1'):
            absolute = True
        return {'kind': 'umi', 'profile': 'umi_' + dataset, 'has_pose': bool(contract.get('eef_sides')),
                'coordinates': {'position_unit': 'unverified',
                                'target_frame': 'per_arm_first_state' if first_state else 'saved UMI reference',
                                **({'common_base_known': False} if first_state else {})},
                'action_kind': 'absolute_pose' if absolute else None,
                'pose_encoding': 'xyz_rotation_first_two_rows',
                'source': 'single_arm' if len(contract.get('eef_sides', [])) == 1 else 'dual_arm',
                'robot': dataset, 'configuration': group.get('processing', '')}
    return {'kind': 'legacy', 'profile': details.get('profile', ''),
            'coordinates': contract.get('coordinates', {}), 'action_kind': contract.get('action_kind'),
            'has_pose': None, 'source': '', 'robot': details.get('profile', ''),
            'configuration': group.get('processing', '')}


def alignment_policy(group, policies):
    key = group['dataset'] + '/' + group.get('directory', '')
    policy = policies.get(key)
    if policy is None:
        return None
    if policy.get('mapping_sha256') != digest(group['contract']):
        raise ValueError('alignment policy belongs to another mapping version: ' + key)
    allowed = {'HABIT': 'chassis', 'realomin_10kh': 'episode_first_state'}
    if (policy.get('mode') != allowed.get(group['dataset']) or not policy.get('evidence')
            or policy.get('position_unit_assumption') != 'meter'):
        raise ValueError('invalid explicit alignment policy: ' + key)
    return policy


def group_label(key, group):
    description = describe(group)
    parts = [group['dataset'], description['source'], description['robot']]
    return ' / '.join(dict.fromkeys(p for p in parts if p)) + ' / ' + key.rsplit('__', 1)[-1]


def episode_review(review, valid_pose_frames):
    if valid_pose_frames is not None and not any(valid_pose_frames.values()):
        return {'status': 'no_eef', 'reason': 'episode has no valid EEF pose; video and saved values only'}
    return review
