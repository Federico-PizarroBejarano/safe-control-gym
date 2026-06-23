'''This script tests the MPSC safety filter implementation.'''

import os
import pickle
import shutil
import sys
from functools import partial

import numpy as np

from experiments.mpsc.plotting_results import plot_trajectories
from safe_control_gym.envs.benchmark_env import Cost, Environment, Task
from safe_control_gym.experiments.base_experiment import BaseExperiment, MetricExtractor
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function, get_discrete_derivative
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def parse_model_arg():
    '''Parse --model=NAME from command-line arguments.'''
    for arg in sys.argv:
        if arg.startswith('--model='):
            return arg.split('=', 1)[1]
    return None


def run(plot=True, training=False, n_episodes=1, n_steps=None, curr_path='.', init_state=None, model='none'):
    '''Main function to run MPSC experiments.

    Args:
        plot (bool): Whether to plot the results.
        training (bool): Whether to train the MPSC or load pre-trained values.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.
        init_state (np.ndarray): Optionally can add a different initial state.
        model (str): Optional model name to use for RL agent.

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
    config.task_config['done_on_violation'] = False
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
        checkpoint_dir = os.path.join(
            curr_path, 'models', 'rl_models', system, task, config.algo, model)
        seed = config.task_config.get('seed', None)
        if seed is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, f'seed_{seed}')
        checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pt')
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Checkpoint not found at {checkpoint_path}. '
                f'Expected models/rl_models/{system}/{task}/{config.algo}/{model}/'
                f'seed_{seed}/model_best.pt when using seed subdirs from train_model.sbatch.')
        ctrl.load(checkpoint_path)

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

    if config.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl
        safety_filter.cost_function.output_dir = curr_path
        if config.algo == 'pid':
            ctrl.save(f'{curr_path}/temp-data/saved_controller_prev.npy')

    if training is True:
        train_env = env_func(randomized_init=True,
                             init_state=None,
                             cost='quadratic',
                             normalized_rl_action_space=False,
                             disturbance=None,
                             )
        safety_filter.learn(env=train_env)
        safety_filter.save(path=f'{curr_path}/models/mpsc_parameters/{config.safety_filter}_{system}.pkl')
    else:
        safety_filter.load(path=f'{curr_path}/models/mpsc_parameters/{config.safety_filter}_{system}.pkl')

    # Run with safety filter
    experiment = BaseExperiment(env, ctrl, safety_filter=safety_filter)
    cert_results, cert_metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    experiment.close()
    safety_filter.close()

    elapsed_time_uncert = np.array(uncert_results['timestamp'])[:, -1] - np.array(uncert_results['timestamp'])[:, 0]
    elapsed_time_cert = np.array(cert_results['timestamp'])[:, -1] - np.array(cert_results['timestamp'])[:, 0]

    mpsc_results = cert_results['safety_filter_data']
    corrections = mpsc_results['correction'][0] * 10.0 > np.linalg.norm(cert_results['current_physical_action'][0] - safety_filter.U_EQ[0], axis=1)
    corrections = np.append(corrections, False)

    print('Total Uncertified Time (s):', np.mean(elapsed_time_uncert))
    print('Total Certified Time (s):', np.mean(elapsed_time_cert))
    print('Number of Corrections:', np.sum(corrections))
    print('Sum of Corrections:', np.linalg.norm(mpsc_results['correction'][0]))
    print('Max Correction:', np.max(np.abs(mpsc_results['correction'][0])))
    print('Number of Feasible Iterations:', np.sum(mpsc_results['feasible'][0]))
    print('Total Number of Iterations:', uncert_metrics['average_length'])
    print('Total Number of Certified Iterations:', cert_metrics['average_length'])
    print('Number of Violations:', uncert_metrics['average_constraint_violation'])
    print('Number of Certified Violations:', cert_metrics['average_constraint_violation'])
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
    config.sf_config.cost_function = 'one_step_cost'

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    env_func = partial(make,
                       config.task,
                       **config.task_config)

    generator_env = env_func(init_state=None, randomized_init=True)

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                         env_func,
                         **config.sf_config)
    safety_filter.reset()

    safety_filter.load(path=f'./models/mpsc_parameters/{config.safety_filter}_{system}.pkl')

    starting_points = []
    nx = generator_env.symbolic.nx
    physical_action = generator_env.symbolic.U_EQ

    while len(starting_points) < num_points:
        init_state, info = generator_env.reset()
        unextended_obs = np.squeeze(init_state)[:nx]
        safety_filter.reset_before_run()
        _, success = safety_filter.certify_action(unextended_obs, physical_action, info)
        if not success and safety_filter.acados_needs_rebuild:
            safety_filter.recover_acados_solver(rebuild=True)
            _, success = safety_filter.certify_action(unextended_obs, physical_action, info)
        elif np.all(safety_filter.slack_prev < 1e-4):
            starting_points += [init_state]

    generator_env.close()
    safety_filter.close()

    print(starting_points)
    np.save(f'./models/starting_points/{system}/starting_points_{system}_{task}.npy', starting_points)


def run_multiple_models(plot=True, model=None):
    '''Runs all models at every saved starting point.

    Args:
        plot (bool): Whether to plot the results.
        model (str): A specific model to evaluate.
    '''

    fac = ConfigFactory()
    config = fac.merge()

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    starting_points = np.load(f'./models/starting_points/{system}/starting_points_{system}_{task}.npy')

    if model is None:
        all_models = os.listdir(f'./models/rl_models/{system}/{task}/{config.algo}/')
    else:
        all_models = [model]

    for model in all_models:
        for i in range(starting_points.shape[0]):
            init_state = starting_points[i, :]
            X_GOAL, uncert_results, _, cert_results, _ = run(plot=plot, training=False, n_episodes=1, n_steps=None, curr_path='.', init_state=init_state, model=model)
            if i == 0:
                all_uncert_results, all_cert_results = uncert_results, cert_results
            else:
                for key in all_cert_results.keys():
                    if key in ['controller_data', 'safety_filter_data']:
                        for sec_key in cert_results[key].keys():
                            if key != 'safety_filter_data':
                                all_uncert_results[key][sec_key].append(uncert_results[key][sec_key][0])
                            all_cert_results[key][sec_key].append(cert_results[key][sec_key][0])
                    else:
                        all_uncert_results[key].append(uncert_results[key][0])
                        all_cert_results[key].append(cert_results[key][0])

        met = MetricExtractor()
        uncert_metrics = met.compute_metrics(data=all_uncert_results)
        cert_metrics = met.compute_metrics(data=all_cert_results)

        all_results = {'uncert_results': all_uncert_results,
                       'uncert_metrics': uncert_metrics,
                       'cert_results': all_cert_results,
                       'cert_metrics': cert_metrics,
                       'config': config,
                       'X_GOAL': X_GOAL}

        os.makedirs(f'./results_mpsc/{system}/{task}/{config.algo}/results_{system}_{task}_{config.algo}_{model}/', exist_ok=True)
        with open(f'./results_mpsc/{system}/{task}/{config.algo}/results_{system}_{task}_{config.algo}_{model}/seed_{config.task_config.seed}.pkl', 'wb') as f:
            pickle.dump(all_results, f)


if __name__ == '__main__':
    # run()
    # determine_feasible_starting_points(num_points=100)
    model = parse_model_arg()
    if model is not None and model != 'none':
        run(plot=True, training=False, n_episodes=1, model=model)
    else:
        run_multiple_models(plot=False, model=model if model != 'none' else None)
