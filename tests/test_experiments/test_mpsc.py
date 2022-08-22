import sys
import pytest

from experiments.mpsc.mpsc_experiment import run

@pytest.mark.parametrize('SYS',             ['cartpole', 'quadrotor_2D'])
@pytest.mark.parametrize('TASK',            ['stab', 'track'])
@pytest.mark.parametrize('ALGO',            ['lqr', 'pid', 'ppo', 'sac'])
@pytest.mark.parametrize('SAFETY_FILTER',   ['linear_mpsc', 'nl_mpsc'])
@pytest.mark.parametrize('MPSC_COST',       ['one_step_cost', 'lqr_cost', 'precomputed_cost', 'learned_cost'])
def test_mpsc(SYS, TASK, ALGO, SAFETY_FILTER, MPSC_COST):
    if SYS == 'cartpole' and ALGO == 'pid':
        return True
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
    run(plot=False, training=False, n_episodes=None, n_steps=2, curr_path='./experiments/mpsc')
