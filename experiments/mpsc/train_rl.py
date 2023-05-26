'''Template training/plotting/testing script.'''

import os
import shutil
from functools import partial

import munch
import yaml

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_device_from_config, set_seed_from_config
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.envs.benchmark_env import Task, Environment
from safe_control_gym.utils.plotting import plot_from_logs


def train():
    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    config.algo_config['training'] = True

    if os.path.isdir(config.output_dir):
        cont = input(f'Overwrite {config.output_dir} [y/n]: ')
        if cont.lower() == 'y':
            shutil.rmtree(config.output_dir, ignore_errors=True)
        else:
            return

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    set_seed_from_config(config)
    set_device_from_config(config)

    # Define function to create task/env.
    env_func = partial(make,
                       config.task,
                       output_dir=config.output_dir,
                       **config.task_config
                       )

    # Create the controller/control_agent.
    ctrl = make(config.algo,
                env_func,
                checkpoint_path=os.path.join(config.output_dir, 'model_latest.pt'),
                output_dir=config.output_dir,
                use_gpu=config.use_gpu,
                seed=config.seed,
                **config.algo_config)
    ctrl.reset()

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                         env_func,
                         **config.sf_config)
    safety_filter.reset()

    if config.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl
        safety_filter.cost_function.output_dir = '.'
        if config.algo == 'pid':
            ctrl.save('./temp-data/saved_controller_prev.npy')

    safety_filter.load(path=f'./models/mpsc_parameters/{config.safety_filter}_{system}_{task}.pkl')

    ctrl.safety_filter = safety_filter

    # Training.
    ctrl.learn()
    ctrl.close()
    print('Training done.')

    with open(os.path.join(config.output_dir, 'config.yaml'), 'w', encoding='UTF-8') as file:
        yaml.dump(munch.unmunchify(config), file, default_flow_style=False)

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
