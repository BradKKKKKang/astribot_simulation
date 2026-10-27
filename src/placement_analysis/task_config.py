"""Placement input adapter retaining the recorded flat settings contract."""
from src.simulation_config.placement import DEFAULT_DETECTOR_CONFIG, EVENT_TYPES, validate_task_config


def load_task_config(path, *, require_ready=True, validate_identification=False):
    from src.simulation_config.experiment_config import resolve_experiment
    resolved = resolve_experiment(path, require_ready=require_ready)
    if resolved['kind'] != 'placement_test':
        raise ValueError('replay/extract/identify require a placement_test experiment')
    return validate_task_config(resolved['settings'], require_ready=require_ready,
                                validate_identification=validate_identification)
