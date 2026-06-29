#!/usr/bin/env python3
'''Train PPO with NL-MPSC via the adversarial harness (backward-compatible entry).'''

import copy
import os
import shutil
import time
from functools import partial

import munch
import yaml

from experiments.adversarial.adversarial_nominal_ppo import AdversarialNominalPPO
from experiments.adversarial.config_utils import require_task_constraints, strip_disturbances
from safe_control_gym.controllers.ppo.ppo import PPO
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_device_from_config, set_seed_from_config

_CONTROLLERS = {'nominal': PPO, 'adversary': AdversarialNominalPPO}


def train():
    config = ConfigFactory().merge()
    config.algo_config['training'] = True
    strip_disturbances(config)
    require_task_constraints(config)

    shutil.rmtree(config.output_dir, ignore_errors=True)
    mkdirs(config.output_dir)

    set_seed_from_config(config)
    set_device_from_config(config)

    env_func = partial(make, config.task, output_dir=config.output_dir, **config.task_config)

    algo_cfg = copy.deepcopy(munch.unmunchify(config.algo_config))
    strategy = algo_cfg.pop('adversary_strategy', 'adversary')
    if strategy not in _CONTROLLERS:
        raise ValueError(f'Unknown adversary_strategy {strategy!r}; use {_CONTROLLERS.keys()}')

    safety_filter = make(config.safety_filter, env_func, **config.sf_config)
    safety_filter.reset()

    ctrl = _CONTROLLERS[strategy](
        env_func,
        checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
        output_dir=config.output_dir,
        use_gpu=config.use_gpu,
        seed=config.seed,
        **algo_cfg,
    )
    ctrl.reset()
    ctrl.safety_filter = safety_filter

    with open(os.path.join(config.output_dir, 'config.yaml'), 'w', encoding='UTF-8') as f:
        yaml.dump(munch.unmunchify(config), f, default_flow_style=False)

    t0 = time.time()
    ctrl.learn()
    ctrl.close()
    with open(os.path.join(config.output_dir, 'training_time.txt'), 'w', encoding='UTF-8') as f:
        f.write(f'Total training time: {time.time() - t0:.1f}s')
    print('Training done.')

    make_plots(config)


def make_plots(config):
    '''Produces plots for logged stats during training.
    Usage
        * use with `--func plot` and `--restore {dir_path}` where `dir_path` is
            the experiment folder containing the logs.
        * save figures under `dir_path/plots/`.
    '''
    log_dir = os.path.join(config.output_dir, 'logs')
    plot_dir = os.path.join(config.output_dir, 'plots')
    mkdirs(plot_dir)
    plot_from_logs(log_dir, plot_dir, window=3)
    print('Plotting done.')


if __name__ == '__main__':
    train()
