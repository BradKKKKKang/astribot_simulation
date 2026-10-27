"""Per-hand gripper transitions, TCP stop/press candidates and startup annotations."""
import math

import numpy as np


from src.simulation_config.candidates import DetectorConfig
from src.simulation_config.poses import quaternion_multiply, rotate, compose_pose, relative_pose


def detect_transitions(times, openings, config=DetectorConfig()):
    """Return one grasp/release per confirmed transition; initial state is silent.

    Values are SDK percentages, normalized here so 100 means open. Grasp uses
    the first closed sample, or the configured closure onset retrospectively;
    release uses departure from the closed band. A
    sustained target state confirms the event retrospectively. Missing-time
    gaps reset the state rather than inventing an event across the gap.
    """
    times, openings = np.asarray(times, dtype=float), np.asarray(openings, dtype=float)
    if times.ndim != 1 or times.shape != openings.shape or not np.isfinite(times).all():
        raise ValueError("times/openings must be equal-length finite vectors")
    if not np.isfinite(openings).all() or np.any((openings < -1e-3) | (openings > 100.001)):
        raise ValueError("gripper samples must be percentages in [0, 100]")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must increase strictly")
    if not config.larger_is_open:
        openings = 100 - openings
    state, candidate, candidate_at, departure = None, None, None, None
    grasp_onset = None
    events = []
    for i, value in enumerate(openings):
        if i and times[i] - times[i - 1] > config.max_gap_s:
            state, candidate, candidate_at, departure = None, None, None, None
            grasp_onset = None
        if state == 'open' and config.grasp_onset_threshold is not None:
            if 100 - value < config.grasp_onset_threshold:
                grasp_onset = None
            elif grasp_onset is None and i and 100 - openings[i - 1] < config.grasp_onset_threshold:
                grasp_onset = i
        label = ("closed" if value <= config.closed_threshold else
                 "open" if value >= config.open_threshold else None)
        if label == state and state is not None:
            departure = None
        elif state is not None and departure is None:
            departure = i
        if label is None or label == state:
            candidate, candidate_at = None, None
            continue
        if candidate != label:
            candidate, candidate_at = label, i
        if times[i] - times[candidate_at] + 1e-9 < config.confirm_duration_s:
            continue
        if state is not None:
            grasp = label == "closed"
            index = candidate_at if grasp else departure
            if grasp and grasp_onset is not None:
                index = grasp_onset
            events.append({"event_type": "grasp" if grasp else "release",
                           "frame_index": int(index),
                           "transition_start_index": int(grasp_onset if grasp and grasp_onset is not None else departure),
                           "transition_end_index": int(candidate_at),
                           "confirmed_index": i})
        state, candidate, candidate_at, departure = label, None, None, None
        grasp_onset = None
    return events


def motion_metrics(times, positions, config=DetectorConfig(), *, indices=None):
    """Physical-time derivatives, optionally at selected frames only.

    Selected frames use exactly the same local windows, including boundaries
    and gaps, as the full trajectory calculation. Other entries remain NaN.
    """
    times, positions = np.asarray(times, float), np.asarray(positions, float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('timestamps must be finite and strictly increasing')
    if positions.shape != (len(times), 3) or not np.isfinite(positions).all():
        raise ValueError("TCP positions must have finite shape (N, 3)")
    speed, acceleration = np.full(len(times), np.nan), np.full(len(times), np.nan)
    selected = None
    if indices is not None:
        selected = np.asarray(indices)
        if selected.ndim != 1 or (selected.size and (not np.issubdtype(selected.dtype, np.integer)
                                                     or np.any(selected < 0) or np.any(selected >= len(times)))):
            raise ValueError("metric indices must be valid frame indices")
        selected = set(selected.tolist())
    segments = np.split(np.arange(len(times)), np.flatnonzero(np.diff(times) > config.max_gap_s) + 1)
    for indices in segments:
        if len(indices) < 3:
            continue
        points = positions[indices]
        radius = max(1, config.smoothing_samples // 2)
        window = min(len(indices), 2 * radius + 1)
        for i, index in enumerate(indices):
            if selected is not None and index not in selected:
                continue
            start = min(max(0, i - radius), len(indices) - window)
            local_t = times[indices[start:start + window]] - times[index]
            # A local quadratic fit gives derivatives at the actual event time,
            # including boundaries, without fake padding or wall-clock timing.
            coefficients = np.linalg.lstsq(
                np.c_[np.ones(window), local_t, local_t**2],
                points[start:start + window], rcond=None)[0]
            speed[index] = np.linalg.norm(coefficients[1])
            acceleration[index] = np.linalg.norm(2 * coefficients[2])
    return speed, acceleration


def detect_stops(times, positions, speed, acceleration, config=DetectorConfig()):
    """One candidate per sustained stop, including initial idle for audit.

    Both speed AND acceleration must stay low. After an emitted stop, brief
    threshold noise cannot create another: rearm requires sustained motion
    and displacement. Gaps break windows and reset the local state.
    """
    times, positions = np.asarray(times, float), np.asarray(positions, float)
    speed, acceleration = np.asarray(speed, float), np.asarray(acceleration, float)
    if (times.ndim != 1 or positions.shape != (len(times), 3)
            or speed.shape != times.shape or acceleration.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(positions).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('invalid stop input shapes, times or positions')
    low = (np.isfinite(speed) & np.isfinite(acceleration)
           & (speed <= config.speed_threshold_m_s)
           & (acceleration <= config.acceleration_threshold_m_s2))
    events = []
    for segment in np.split(np.arange(len(times)), np.flatnonzero(np.diff(times) > config.max_gap_s) + 1):
        if not len(segment):
            continue
        armed, last_position, moving_start, run_start = True, None, None, None

        def emit(end):
            nonlocal armed, last_position
            if run_start is None or not armed or times[end] - times[run_start] + 1e-9 < config.dwell_duration_s:
                return
            middle = (times[run_start] + times[end]) / 2
            index = run_start + int(np.argmin(np.abs(times[run_start:end+1] - middle)))
            confirmed = run_start + int(np.searchsorted(times[run_start:end+1],
                                                        times[run_start] + config.dwell_duration_s - 1e-9))
            events.append({'event_type': 'dwell', 'frame_index': index,
                           'transition_start_index': run_start, 'transition_end_index': end,
                           'confirmed_index': confirmed})
            armed, last_position = False, positions[index].copy()

        for index in segment:
            if low[index]:
                moving_start = None
                if run_start is None:
                    run_start = int(index)
            else:
                emit(int(index) - 1)
                run_start = None
                if np.isfinite(speed[index]) and speed[index] > config.speed_threshold_m_s:
                    if moving_start is None:
                        moving_start = int(index)
                    if (last_position is not None
                            and times[index] - times[moving_start] + 1e-9 >= config.motion_rearm_duration_s
                            and np.linalg.norm(positions[index] - last_position) >= config.motion_rearm_distance_m):
                        armed = True
                else:
                    moving_start = None
        emit(int(segment[-1]))
    return events


def smooth_press_positions(times, positions, config=DetectorConfig()):
    """Centered physical-time mean, segmented at missing feedback intervals.

    Unlike a fixed frame count, the time support is stable across capture rates.
    This offline smoother never fills missing samples or crosses a time gap.
    """
    times, positions = np.asarray(times, float), np.asarray(positions, float)
    if (times.ndim != 1 or positions.shape != (len(times), 3)
            or not np.isfinite(times).all() or not np.isfinite(positions).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('invalid press timestamps or TCP positions')
    result = np.empty_like(positions)
    for segment in np.split(np.arange(len(times)), np.flatnonzero(np.diff(times) > config.max_gap_s) + 1):
        if not len(segment):
            continue
        local_t = times[segment]
        left = np.searchsorted(local_t, local_t - config.press_smoothing_s / 2)
        right = np.searchsorted(local_t, local_t + config.press_smoothing_s / 2, side='right')
        sums = np.vstack([np.zeros(3), np.cumsum(positions[segment], axis=0)])
        result[segment] = (sums[right] - sums[left]) / (right - left)[:, None]
    return result


def detect_presses(times, positions, openings, config=DetectorConfig()):
    """Vertical down/bottom/up candidates; contact is not observed.

    A trough must have sufficient descent AND subsequent retraction, limited
    lateral displacement and, optionally, a closed gripper throughout. The
    deepest smoothed sample represents the stroke; its original measured pose
    is retained by the recorder. Confirmation waits for retraction. No episode
    timestamps or desired number of presses enter detection.
    """
    times, openings = np.asarray(times, float), np.asarray(openings, float)
    points = smooth_press_positions(times, positions, config)
    if (openings.shape != times.shape or not np.isfinite(openings).all()
            or np.any((openings < 0) | (openings > 100))):
        raise ValueError('press gripper samples must be percentages in [0, 100]')
    normalized = openings if config.larger_is_open else 100 - openings
    events = []
    for segment in np.split(np.arange(len(times)), np.flatnonzero(np.diff(times) > config.max_gap_s) + 1):
        last_end = -1
        for index in segment[1:-1]:
            z = points[index, 2]
            if index <= last_end or not (z <= points[index - 1, 2] and z < points[index + 1, 2]):
                continue
            before = segment[(times[segment] >= times[index] - config.press_context_s) & (segment < index)]
            after = segment[(times[segment] <= times[index] + config.press_context_s) & (segment > index)]
            down = before[points[before, 2] >= z + config.press_min_depth_m]
            up = after[points[after, 2] >= z + config.press_min_retract_m]
            if not len(down) or not len(up):
                continue
            start, end = int(down[-1]), int(up[0])
            # Reject shallow ripples inside a deeper stroke, without merging
            # two real strokes that are separated by a sufficient lift.
            if (np.min(points[start:end + 1, 2]) < z - 1e-10
                    or min(times[index] - times[start], times[end] - times[index]) < config.press_min_phase_s):
                continue
            lateral = float(np.max(np.linalg.norm(points[start:end + 1, :2] - points[index, :2], axis=1)))
            if lateral > config.press_max_lateral_m:
                continue
            if config.press_require_closed and np.any(normalized[start:end + 1] > config.closed_threshold):
                continue
            events.append({'event_type': 'press', 'frame_index': int(index),
                           'transition_start_index': start, 'transition_end_index': end,
                           'confirmed_index': end,
                           'press_depth_m': float(points[start, 2] - z),
                           'press_retract_m': float(points[end, 2] - z),
                           'press_lateral_m': lateral})
            last_end = end
    return events


def initial_phase_mask(times, positions_base, config=DetectorConfig()):
    """Latch each hand active after sustained departure from its initial TCP.

    A return to the same location after departure stays active. The departure
    is applied retrospectively to the start of the confirmed excursion.
    """
    times, positions = np.asarray(times, float), np.asarray(positions_base, float)
    if (times.ndim != 1 or positions.shape != (len(times), 3)
            or not np.isfinite(times).all() or not np.isfinite(positions).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('invalid startup positions or timestamps')
    if not len(times):
        return np.ones(0, dtype=bool), None, np.zeros(0)
    distance = np.linalg.norm(positions - positions[0], axis=1)
    start, departure = None, None
    for i, value in enumerate(distance):
        if i and times[i] - times[i-1] > config.max_gap_s:
            start = None
        if value < config.initial_radius_m:
            start = None
        else:
            if start is None:
                start = i
            if times[i] - times[start] + 1e-9 >= config.initial_departure_duration_s:
                departure = start
                break
    mask = np.ones(len(times), dtype=bool)
    if departure is not None:
        mask[departure:] = False
    return mask, departure, distance
