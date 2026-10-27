"""Candidate detector configuration; independent of runtime event extraction."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class DetectorConfig:
    type: str = 'gripper'
    closed_threshold: float = 25.0
    open_threshold: float = 75.0
    grasp_onset_threshold: float = None
    larger_is_open: bool = False
    confirm_duration_s: float = 0.08
    max_gap_s: float = 0.1
    speed_threshold_m_s: float = 0.02
    acceleration_threshold_m_s2: float = 0.2
    smoothing_samples: int = 5
    dwell_duration_s: float = 0.2
    motion_rearm_duration_s: float = 0.08
    motion_rearm_distance_m: float = 0.01
    coincidence_tolerance_s: float = 0.15
    initial_radius_m: float = 0.05
    initial_departure_duration_s: float = 0.15
    initial_ignore_s: float = 1.0
    press_smoothing_s: float = 0.2
    press_context_s: float = 1.5
    press_min_depth_m: float = 0.02
    press_min_retract_m: float = 0.015
    press_max_lateral_m: float = 0.025
    press_min_phase_s: float = 0.08
    press_require_closed: bool = True

    def __post_init__(self):
        numbers = [self.closed_threshold, self.open_threshold, self.confirm_duration_s,
                   self.max_gap_s, self.speed_threshold_m_s, self.acceleration_threshold_m_s2,
                   self.dwell_duration_s, self.motion_rearm_duration_s, self.motion_rearm_distance_m,
                   self.coincidence_tolerance_s, self.initial_radius_m,
                   self.initial_departure_duration_s, self.initial_ignore_s]
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("detector settings must be finite")
        if not 0 <= self.closed_threshold < self.open_threshold <= 100:
            raise ValueError("require 0 <= closed_threshold < open_threshold <= 100")
        if self.confirm_duration_s < 0 or self.max_gap_s <= 0:
            raise ValueError("invalid confirmation duration or max gap")
        if self.speed_threshold_m_s < 0 or self.acceleration_threshold_m_s2 < 0:
            raise ValueError("motion thresholds must be nonnegative")
        if not isinstance(self.smoothing_samples, int) or self.smoothing_samples < 1:
            raise ValueError("smoothing_samples must be a positive integer")
        if min(self.dwell_duration_s, self.motion_rearm_duration_s,
               self.motion_rearm_distance_m, self.initial_radius_m,
               self.initial_departure_duration_s) <= 0:
            raise ValueError('stop/rearm/departure durations and distances must be positive')
        if self.coincidence_tolerance_s < 0 or self.initial_ignore_s < 0:
            raise ValueError('coincidence tolerance and initial ignore time must be nonnegative')
        if self.type not in ('gripper', 'press'):
            raise ValueError('type must be gripper or press')
        if self.grasp_onset_threshold is not None and not (
                math.isfinite(self.grasp_onset_threshold)
                and 0 < self.grasp_onset_threshold < 100 - self.open_threshold):
            raise ValueError('grasp_onset_threshold must be within the open band (normalized closure percent)')
        press = [self.press_smoothing_s, self.press_context_s, self.press_min_depth_m,
                 self.press_min_retract_m, self.press_max_lateral_m, self.press_min_phase_s]
        if not all(math.isfinite(value) and value > 0 for value in press):
            raise ValueError('press durations and distances must be finite and positive')
        if self.press_min_phase_s >= self.press_context_s:
            raise ValueError('press phase must be shorter than context')
        if not isinstance(self.press_require_closed, bool):
            raise ValueError('press_require_closed must be boolean')
