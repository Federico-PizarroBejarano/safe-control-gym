import sys
import pytest

import numpy as np

from experiments.mpsc.mpsc_experiment import run

@pytest.mark.parametrize('SYS',             ['cartpole', 'quadrotor_2D', 'quadrotor_3D'])
@pytest.mark.parametrize('TASK',            ['stab', 'track'])
@pytest.mark.parametrize('ALGO',            ['lqr', 'pid', 'ppo', 'sac'])
@pytest.mark.parametrize('SAFETY_FILTER',   ['linear_mpsc', 'nl_mpsc'])
@pytest.mark.parametrize('MPSC_COST',       ['one_step_cost', 'lqr_cost', 'precomputed_cost', 'learned_cost'])
def test_mpsc(SYS, TASK, ALGO, SAFETY_FILTER, MPSC_COST):
    if SYS == 'cartpole' and ALGO == 'pid':
        pytest.skip('PID is designed for quadrotors and does not function for the carpole system.')
    if SYS == 'quadrotor_3D' and SAFETY_FILTER == 'linear_mpsc':
        pytest.skip('Linear MPSC currently does not function with the 3D quadrotor.')
    SYS_NAME = SYS if SYS == 'cartpole' else 'quadrotor'
    sys.argv[1:] = [
        '--task', SYS_NAME,
        '--algo', ALGO,
        '--safety_filter', SAFETY_FILTER,
        '--overrides',
            f'./experiments/mpsc/config_overrides/{SYS}/{SYS}_{TASK}.yaml',
            f'./experiments/mpsc/config_overrides/{SYS}/{ALGO}_{SYS}.yaml',
            f'./experiments/mpsc/config_overrides/{SYS}/{SAFETY_FILTER}_{SYS}.yaml',
        '--kv_overrides', f'sf_config.cost_function={MPSC_COST}'
        ]
    _, _, cert_results, cert_metrics = run(plot=False, training=False, n_episodes=None, n_steps=2, curr_path='./experiments/mpsc')

    mpsc_results = cert_results['safety_filter_data'][0]
    feasible_iterations = np.sum(mpsc_results['feasible'][0])

    assert cert_metrics['average_constraint_violation'] == 0
    assert cert_metrics['average_length'] == feasible_iterations
