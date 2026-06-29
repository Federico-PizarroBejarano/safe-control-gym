'''Shared config checks for experiments/adversarial entry scripts.'''


def strip_disturbances(config):
    tc = getattr(config, 'task_config', None)
    if tc is None:
        return
    tc.disturbances = None
    tc.adversary_disturbance = None


def require_task_constraints(config):
    c = getattr(getattr(config, 'task_config', None), 'constraints', None)
    if c is None:
        raise ValueError(
            'task_config.constraints is required for NL-MPSC. '
            'See config_overrides/cartpole/cartpole_track.yaml.')
