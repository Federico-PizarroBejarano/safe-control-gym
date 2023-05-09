'''This script tests the MPSC safety filter implementation.'''

import pickle
import shutil
from functools import partial

import numpy as np

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.envs.benchmark_env import Task, Cost, Environment
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function, high_frequency_content, get_discrete_derivative, approximate_LQR_gain
from experiments.mpsc.plotting_results import plot_trajectories


reachable_state_randomization = {
    'cartpole': {
        'init_x': {
            'distrib': 'uniform',
            'low': -2,
            'high': 2},
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -2,
            'high': 2},
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.16,
            'high': 0.16},
        'init_theta_dot': {
            'distrib': 'uniform',
            'low': -1,
            'high': 1}
    },
    'quadrotor_2D': {
        'init_x': {
            'distrib': 'uniform',
            'low': -2,
            'high': 2},
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -1,
            'high': 1},
        'init_z': {
            'distrib': 'uniform',
            'low': 1,
            'high': 2},
        'init_z_dot': {
            'distrib': 'uniform',
            'low': -1,
            'high': 1},
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.2,
            'high': 0.2},
        'init_theta_dot': {
            'distrib': 'uniform',
            'low': -1.5,
            'high': 1.5}
    },
    'quadrotor_3D': {
        'init_x': {
            'distrib': 'uniform',
            'low': -2,
            'high': 2},
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -1,
            'high': 1},
        'init_y': {
            'distrib': 'uniform',
            'low': -2,
            'high': 2},
        'init_y_dot': {
            'distrib': 'uniform',
            'low': -1,
            'high': 1},
        'init_z': {
            'distrib': 'uniform',
            'low': 1,
            'high': 2},
        'init_z_dot': {
            'distrib': 'uniform',
            'low': -1,
            'high': 1},
        'init_phi': {
            'distrib': 'uniform',
            'low': -0.2,
            'high': 0.2},
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.2,
            'high': 0.2},
        'init_psi': {
            'distrib': 'uniform',
            'low': -0.2,
            'high': 0.2},
        'init_p': {
            'distrib': 'uniform',
            'low': -1,
            'high': 1},
        'init_q': {
            'distrib': 'uniform',
            'low': -1,
            'high': 1},
        'init_r': {
            'distrib': 'uniform',
            'low': -1,
            'high': 1}
    },
}

regularization_parameters = {
    'cartpole': {
        'stab': {
            'lqr': 0.0,
            'ppo': 0.125,
            'sac': 0.07,
        },
        'track': {
            'lqr': 0.0,
            'ppo': 0.01,
            'sac': 1000.0,
        },
    },
    'quadrotor_2D': {
        'stab': {
            'lqr': 0.0,
            'pid': 2.5,
            'ppo': 1.0,
            'sac': 1.0,
        },
        'track': {
            'lqr': 0.0,
            'pid': 10000.0,
            'ppo': 10000.0,
            'sac': 100.0,
        },
    },
    'quadrotor_3D': {
        'stab': {
            'lqr': 15.0,
            'pid': 10000.0,
            'ppo': 100.0,
            'sac': 100.0,
        },
        'track': {
            'lqr': 1000000.0,
            'pid': 10000.0,
            'ppo': 100.0,
            'sac': 100.0,
        },
    }
}


def run(plot=True, training=False, n_episodes=1, n_steps=None, curr_path='.', init_state=None):
    '''Main function to run MPSC experiments.

    Args:
        plot (bool): Whether to plot the results.
        training (bool): Whether to train the MPSC or load pre-trained values.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.
        init_state (np.ndarray): Optionally can add a different initial state.

    Returns:
        X_GOAL (np.ndarray): The goal (stabilization or reference trajectory) of the experiment.
        uncert_results (dict): The results of the uncertified experiment.
        uncert_metrics (dict): The metrics of the uncertified experiment.
        cert_results (dict): The results of the certified experiment.
        cert_metrics (dict): The metrics of the certified experiment.
    '''

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    config.algo_config['training'] = False
    if init_state is not None:
        config.task_config['init_state'] = init_state
    config.task_config['randomized_init'] = False
    if config.algo in ['ppo', 'sac']:
        config.task_config['cost'] = Cost.RL_REWARD
        config.task_config['normalized_rl_action_space'] = True
    else:
        config.task_config['cost'] = Cost.QUADRATIC
        config.task_config['normalized_rl_action_space'] = False

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func()

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir=curr_path + '/temp')

    if config.algo in ['ppo', 'sac']:
        # Load state_dict from trained.
        ctrl.load(f'{curr_path}/models/rl_models/{config.algo}_model_{system}_{task}.pt')

        # Remove temporary files and directories
        shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)

    # Run without safety filter
    experiment = BaseExperiment(env, ctrl)
    uncert_results, uncert_metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    ctrl.reset()

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                         env_func,
                         **config.sf_config)
    safety_filter.reset()

    if config.sf_config.cost_function == Cost_Function.LQR_COST:
        if config.algo == 'lqr':
            safety_filter.cost_function.gain = ctrl.gain
        else:
            safety_filter.cost_function.gain = approximate_LQR_gain(env, ctrl, config, curr_path)
    elif config.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl
        safety_filter.cost_function.output_dir = curr_path
        if config.algo == 'pid':
            ctrl.save(f'{curr_path}/temp-data/saved_controller_prev.npy')
    elif config.sf_config.cost_function == Cost_Function.LEARNED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl
        safety_filter.cost_function.regularization_const = regularization_parameters[system][task][config.algo]
        safety_filter.cost_function.learn_policy(path=f'{curr_path}/models/trajectories/{system}/{config.algo}_data_{system}_{task}.pkl')

    if training is True:
        train_env = env_func(randomized_init=True,
                             init_state_randomization_info=reachable_state_randomization[system],
                             init_state=None,
                             cost='quadratic',
                             normalized_rl_action_space=False,
                             disturbance=None,
                             )
        safety_filter.learn(env=train_env)
        safety_filter.save(path=f'{curr_path}/models/mpsc_parameters/{config.safety_filter}_{system}_{task}.pkl')
    else:
        safety_filter.load(path=f'{curr_path}/models/mpsc_parameters/{config.safety_filter}_{system}_{task}.pkl')

    if config.sf_config.cost_function == Cost_Function.LEARNED_COST:
        safety_filter.setup_optimizer()

    # Run with safety filter
    experiment = BaseExperiment(env, ctrl, safety_filter=safety_filter)
    cert_results, cert_metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    experiment.close()
    safety_filter.close()

    elapsed_time_uncert = uncert_results['timestamp'][0][-1] - uncert_results['timestamp'][0][0]
    elapsed_time_cert = cert_results['timestamp'][0][-1] - cert_results['timestamp'][0][0]

    mpsc_results = cert_results['safety_filter_data'][0]
    corrections = mpsc_results['correction'][0] * 10.0 > np.linalg.norm(cert_results['current_physical_action'][0] - safety_filter.U_EQ[0], axis=1)
    corrections = np.append(corrections, False)

    print('Total Uncertified (s):', elapsed_time_uncert)
    print('Total Certified Time (s):', elapsed_time_cert)
    print('Number of Corrections:', np.sum(corrections))
    print('Sum of Corrections:', np.linalg.norm(mpsc_results['correction'][0]))
    print('Max Correction:', np.max(np.abs(mpsc_results['correction'][0])))
    print('Number of Feasible Iterations:', np.sum(mpsc_results['feasible'][0]))
    print('Total Number of Iterations:', uncert_metrics['average_length'])
    print('Total Number of Certified Iterations:', cert_metrics['average_length'])
    print('Number of Violations:', uncert_metrics['average_constraint_violation'])
    print('Number of Certified Violations:', cert_metrics['average_constraint_violation'])
    print('HFC Uncertified:', high_frequency_content(uncert_results['current_physical_action'][0], config.task_config.ctrl_freq))
    print('HFC Certified:', high_frequency_content(cert_results['current_physical_action'][0], config.task_config.ctrl_freq))
    derivative = get_discrete_derivative(uncert_results['current_physical_action'][0] - safety_filter.U_EQ[0], config.task_config.ctrl_freq)
    derivative = get_discrete_derivative(derivative, config.task_config.ctrl_freq)
    total_derivatives = np.linalg.norm(derivative, 'fro')
    print('2nd Order RoC Uncert:', total_derivatives)
    derivative = get_discrete_derivative(cert_results['current_physical_action'][0] - safety_filter.U_EQ[0], config.task_config.ctrl_freq)
    derivative = get_discrete_derivative(derivative, config.task_config.ctrl_freq)
    total_derivatives = np.linalg.norm(derivative, 'fro')
    print('2nd Order RoC Cert:', total_derivatives)
    print('RMSE Uncertified:', uncert_metrics['average_rmse'])
    print('RMSE Certified:', cert_metrics['average_rmse'])

    if plot is True:
        plot_trajectories(config, safety_filter.env.X_GOAL, uncert_results, cert_results)

    return env.X_GOAL, uncert_results, uncert_metrics, cert_results, cert_metrics


def determine_feasible_starting_points(num_points=100):
    '''Calculates feasible starting points for a system and task.

    Args:
        num_points (int): The number of points to generate.
    '''

    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    config.algo_config['training'] = False
    if config.algo in ['ppo', 'sac']:
        config.task_config['cost'] = Cost.RL_REWARD
        config.task_config['normalized_rl_action_space'] = True
    else:
        config.task_config['cost'] = Cost.QUADRATIC
        config.task_config['normalized_rl_action_space'] = False

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    env_func = partial(make,
                       config.task,
                       **config.task_config)
    state_randomization = reachable_state_randomization[system]
    if system == 'quadrotor_3D':
        for state in state_randomization.keys():
            if 'x' not in state and 'y' not in state and 'z' not in state:
                state_randomization[state]['low'] = 0.1 * state_randomization[state]['low']
                state_randomization[state]['high'] = 0.1 * state_randomization[state]['high']
            else:
                state_randomization[state]['low'] = 0.5 * state_randomization[state]['low']
                state_randomization[state]['high'] = 0.5 * state_randomization[state]['high']
    generator_env = env_func(init_state=None, randomized_init=True, init_state_randomization_info=state_randomization)

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir='./temp')

    if config.algo in ['ppo', 'sac']:
        # Load state_dict from trained.
        ctrl.load(f'./models/rl_models/{config.algo}_model_{system}_{task}.pt')

        # Remove temporary files and directories
        shutil.rmtree('./temp', ignore_errors=True)

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                         env_func,
                         **config.sf_config)
    safety_filter.reset()

    safety_filter.load(path=f'./models/mpsc_parameters/{config.safety_filter}_{system}_{task}.pkl')

    if config.sf_config.cost_function != Cost_Function.ONE_STEP_COST:
        raise ValueError('Currently starting point generation should only be done with one_step_cost.')

    starting_points = []

    while len(starting_points) < num_points:
        generator_env.reset()
        init_state = generator_env.state
        test_env = env_func(init_state=init_state, randomized_init=False)

        uncert_experiment = BaseExperiment(test_env, ctrl)
        cert_experiment = BaseExperiment(test_env, ctrl, safety_filter=safety_filter)

        _, uncert_metrics = uncert_experiment.run_evaluation(n_episodes=1)
        uncert_experiment.reset()
        cert_results, cert_metrics = cert_experiment.run_evaluation(n_steps=10)
        cert_experiment.reset()
        test_env.close()

        mpsc_results = cert_results['safety_filter_data'][0]

        if cert_metrics['average_length'] == 10 \
                and np.all(mpsc_results['feasible']) \
                and uncert_metrics['average_constraint_violation'] > 5 \
                and uncert_metrics['average_length'] == config.task_config.ctrl_freq * config.task_config.episode_len_sec \
                and cert_metrics['average_constraint_violation'] == 0:
            starting_points.append(cert_results['state'][0][0])

    uncert_experiment.close()
    cert_experiment.close()

    print(starting_points)
    np.save(f'./models/starting_points/{system}/starting_points_{system}_{task}_{config.algo}.npy', starting_points)


def run_multiple(plot=True):
    '''Runs an experiment at every saved starting point.

    Args:
        plot (bool): Whether to plot the results.

    Returns:
        uncert_results (dict): The results of the uncertified experiments.
        uncert_metrics (dict): The metrics of the uncertified experiments.
        cert_results (dict): The results of the certified experiments.
        cert_metrics (dict): The metrics of the certified experiments.
    '''

    fac = ConfigFactory()
    config = fac.merge()

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    starting_points = np.load(f'./models/starting_points/{system}/starting_points_{system}_{task}_{config.algo}.npy')

    for i in range(starting_points.shape[0]):
        init_state = starting_points[i, :]
        X_GOAL, uncert_results, _, cert_results, _ = run(plot=plot, training=False, n_episodes=1, n_steps=None, curr_path='.', init_state=init_state)
        if i == 0:
            all_uncert_results, all_cert_results = uncert_results, cert_results
        else:
            for key in all_cert_results.keys():
                if key in all_uncert_results:
                    all_uncert_results[key].append(uncert_results[key][0])
                all_cert_results[key].append(cert_results[key][0])

    uncert_metrics = BaseExperiment.compute_metrics(all_uncert_results)
    cert_metrics = BaseExperiment.compute_metrics(all_cert_results)

    all_results = {'uncert_results': all_uncert_results,
                   'uncert_metrics': uncert_metrics,
                   'cert_results': all_cert_results,
                   'cert_metrics': cert_metrics,
                   'config': config,
                   'X_GOAL': X_GOAL}

    with open(f'./results_mpsc/{system}/{task}/m{config.sf_config.mpsc_cost_horizon}/results_{system}_{task}_{config.algo}_{config.sf_config.cost_function}_m{config.sf_config.mpsc_cost_horizon}.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    return all_uncert_results, uncert_metrics, all_cert_results, cert_metrics


def run_uncertified_trajectory(n_episodes=10):
    '''Runs and saves several initializations of the uncertified trajectories.

    Args:
        n_episodes (int): The number of episodes to execute.
    '''

    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    config.algo_config['training'] = False
    config.task_config['randomized_init'] = False
    if config.algo in ['ppo', 'sac']:
        config.task_config['cost'] = Cost.RL_REWARD
        config.task_config['normalized_rl_action_space'] = True
    else:
        config.task_config['cost'] = Cost.QUADRATIC
        config.task_config['normalized_rl_action_space'] = False

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func()

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir='./temp')

    if config.algo in ['ppo', 'sac']:
        # Load state_dict from trained.
        ctrl.load(f'./models/rl_models/{config.algo}_model_{system}_{task}.pt')

        # Remove temporary files and directories
        shutil.rmtree('./temp', ignore_errors=True)

    # Run without safety filter
    experiment = BaseExperiment(env, ctrl)
    uncert_results, _ = experiment.run_evaluation(n_episodes=n_episodes)
    experiment.close()

    if len(np.unique([len(uncert_results['state'][i]) for i in range(n_episodes)])) > 1:
        print('[ERROR] - One or more experiments failed. Lengths are: ')
        print([len(uncert_results['state'][i]) for i in range(n_episodes)])
        raise Exception()

    with open(f'./models/trajectories/{system}/{config.algo}_data_{system}_{task}.pkl', 'wb') as f:
        pickle.dump(uncert_results, f)


if __name__ == '__main__':
    run()
    # run_uncertified_trajectory()
    # determine_feasible_starting_points(num_points=10)
    # run_multiple(plot=False)
