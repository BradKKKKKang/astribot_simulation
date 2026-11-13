"""Optional deterministic joint target conditioning in simulation time."""
import math


class JointTargetFilter:
    def __init__(self, settings, joint_names):
        if not isinstance(settings, dict) or set(settings) - set(joint_names):
            raise ValueError('joint_command_filters contains unknown joint names')
        self.settings = {}
        for joint, value in settings.items():
            if not isinstance(value, dict) or set(value) - {'time_constant_s', 'velocity_lookahead_s'}:
                raise ValueError(f'{joint}: invalid joint filter fields')
            values = {key: float(value.get(key, 0.0))
                      for key in ('time_constant_s', 'velocity_lookahead_s')}
            if any(not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError(f'{joint}: filter values must be finite and nonnegative')
            self.settings[joint] = values
        self.reset()

    def reset(self):
        self.previous = {}

    def apply(self, joint, position, velocity, legacy_target, dt):
        if joint not in self.settings:
            return legacy_target
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('joint filter requires positive finite simulation dt')
        settings = self.settings[joint]
        target = position + settings['velocity_lookahead_s'] * velocity
        tau = settings['time_constant_s']
        previous = self.previous.get(joint, target)
        result = target if tau == 0 else previous + (-math.expm1(-dt / tau)) * (target - previous)
        self.previous[joint] = result
        return result
