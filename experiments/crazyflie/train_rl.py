'''Running MPSC using the crazyflie firmware. '''

import os
import shutil
import yaml
import sys
sys.path.insert(0, '/home/federico/GitHub/safe-control-gym')

from functools import partial

import numpy as np
import munch

from experiments.crazyflie.crazyflie_utils import gen_traj
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_device_from_config, set_seed_from_config

try:
    import pycffirmware
except ImportError:
    FIRMWARE_INSTALLED = False
else:
    FIRMWARE_INSTALLED = True
finally:
    print('Module \'cffirmware\' available:', FIRMWARE_INSTALLED)


def train():
    '''The main function creating, running, and closing an environment over N episodes. '''

    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    config.algo_config['training'] = True
    config.task_config['init_state'] = None
    config.task_config['randomized_init'] = True

    shutil.rmtree(config.output_dir, ignore_errors=True)

    set_seed_from_config(config)
    set_device_from_config(config)
    CTRL_FREQ = config.task_config['ctrl_freq']

    env_func = partial(make,
                       config.task,
                       output_dir=config.output_dir,
                       **config.task_config)

    FIRMWARE_FREQ = 500
    config.task_config['ctrl_freq'] = FIRMWARE_FREQ
    env_func_500 = partial(make,
                           config.task,
                           output_dir=config.output_dir,
                           **config.task_config)

    # Create environment.
    firmware_wrapper = make('firmware', env_func_500, FIRMWARE_FREQ, CTRL_FREQ)
    _, _ = firmware_wrapper.reset()
    env = firmware_wrapper.env

    # Create trajectory.
    full_trajectory = gen_traj(CTRL_FREQ, env.EPISODE_LEN_SEC)

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                output_dir=config.output_dir,
                seed=1,
                **config.algo_config)

    ctrl.firmware_wrapper = firmware_wrapper
    ctrl.X_GOAL = full_trajectory
    ctrl.CTRL_DT = 1.0 / CTRL_FREQ
    ctrl.reset()

    # Setup MPSC.
    if config.algo in ['ppo', 'sac']:
        safety_filter = make(config.safety_filter,
                             env_func,
                             **config.sf_config)
        safety_filter.reset()
        safety_filter.load(path=f'./models/mpsc_parameters/{config.safety_filter}_crazyflie_track.pkl')

        safety_filter.env.X_GOAL = full_trajectory
        ctrl.safety_filter = safety_filter

    with open(os.path.join(config.output_dir, 'config.yaml'), 'w', encoding='UTF-8') as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)

    ctrl.learn()
    ctrl.close()
    print('Training done.')

    make_plots(config)


def make_plots(config):
    '''Produces plots for logged stats during training.
    Usage
        * use with `--func plot` and `--restore {dir_path}` where `dir_path` is
            the experiment folder containing the logs.
        * save figures under `dir_path/plots/`.
    '''
    # Define source and target log locations.
    log_dir = os.path.join(config.output_dir, 'logs')
    plot_dir = os.path.join(config.output_dir, 'plots')
    mkdirs(plot_dir)
    plot_from_logs(log_dir, plot_dir, window=3)
    print('Plotting done.')


if __name__ == '__main__':
    train()
