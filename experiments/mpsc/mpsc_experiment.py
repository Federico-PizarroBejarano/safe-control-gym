'''This script tests the MPSC safety filter implementation. '''

import pickle
import shutil
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

from safe_control_gym.experiment import Experiment
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.envs.benchmark_env import Task, Cost, Environment
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function


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
            'ppo': 0.0,
            'sac': 0.0,
        },
        'track': {
            'lqr': 0.0,
            'ppo': 5.0,
            'sac': 200.0,
        },
    },
    'quadrotor_2D': {
        'stab': {
            'lqr': 0.0,
            'pid': 10.0,
            'ppo': 10.0,
            'sac': 10.0,
        },
        'track': {
            'lqr': 0.0,
            'pid': 10.0,
            'ppo': 10.0,
            'sac': 10.0,
        },
    },
    'quadrotor_3D': {
        'stab': {
            'lqr': 0.0,
            'pid': 10.0,
            'ppo': 10.0,
            'sac': 10.0,
        },
        'track': {
            'lqr': 0.0,
            'pid': 10.0,
            'ppo': 10.0,
            'sac': 10.0,
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
        uncert_results (dict): The results of the uncertified experiment.
        uncert_metrics (dict): The metrics of the uncertified experiment.
        cert_results (dict): The results of the certified experiment.
        cert_metrics (dict): The metrics of the certified experiment.
    '''

    # Define arguments.
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

    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func()

    # Setup controller.
    ctrl = make(config.algo,
                    env_func,
                    **config.algo_config,
                    output_dir=curr_path+'/temp')

    if config.algo in ['ppo', 'sac']:
        # Load state_dict from trained.
        ctrl.load(f'{curr_path}/models/rl_models/{config.algo}_model_{system}_{task}.pt')

        # Remove temporary files and directories
        shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)

    # Run without safety filter
    experiment = Experiment(env, ctrl)
    uncert_results, uncert_metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    elapsed_time_uncert = uncert_results['timestamp'][0][-1] - uncert_results['timestamp'][0][0]
    ctrl.reset()

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                env_func,
                **config.sf_config)
    safety_filter.reset()

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

    if config.sf_config.cost_function == Cost_Function.LQR_COST:
        if config.algo == 'lqr':
            q_lin = config.algo_config.q_lqr
            r_lin = config.algo_config.r_lqr
        else:
            q_lin = [1]*safety_filter.model.nx
            r_lin = [0.1]
        safety_filter.cost_function.set_lqr_matrices(q_lin, r_lin)
    elif config.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl
        safety_filter.cost_function.output_dir = curr_path
        if config.algo == 'pid':
            ctrl.save(f'{curr_path}/temp-data/saved_controller_prev.npy')
    elif config.sf_config.cost_function == Cost_Function.LEARNED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl
        safety_filter.cost_function.regularization_const = regularization_parameters[system][task][config.algo]
        safety_filter.cost_function.learn_policy(path=f'{curr_path}/models/trajectories/{config.algo}_data_{system}_{task}.pkl')
        safety_filter.setup_optimizer()

    # Run with safety filter
    experiment = Experiment(env, ctrl, safety_filter=safety_filter)
    cert_results, cert_metrics = experiment.run_evaluation(n_episodes=n_episodes, n_steps=n_steps)
    experiment.close()
    mpsc_results = cert_results['safety_filter_data'][0]
    safety_filter.close()

    elapsed_time_cert = cert_results['timestamp'][0][-1] - cert_results['timestamp'][0][0]

    corrections = mpsc_results['correction'][0] > 1e-4
    corrections = np.append(corrections, False)

    if plot is True:
        if system == Environment.CARTPOLE:
            graph1_1 = 2
            graph1_2 = 3
            graph3_1 = 0
            graph3_2 = 1
        elif system == 'quadrotor_2D':
            graph1_1 = 4
            graph1_2 = 5
            graph3_1 = 0
            graph3_2 = 2
        elif system == 'quadrotor_3D':
            graph1_1 = 6
            graph1_2 = 9
            graph3_1 = 0
            graph3_2 = 4

        _, ax = plt.subplots()
        ax.plot(uncert_results['obs'][0][:, graph1_1], uncert_results['obs'][0][:, graph1_2], 'r--', label='Uncertified')
        ax.plot(cert_results['obs'][0][:,graph1_1], cert_results['obs'][0][:,graph1_2],'.-', label='Certified')
        ax.plot(cert_results['obs'][0][corrections, graph1_1], cert_results['obs'][0][corrections, graph1_2], 'r.', label='Modified')
        ax.scatter(uncert_results['obs'][0][0, graph1_1], uncert_results['obs'][0][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
        theta_constraint = config.task_config['constraints'][0].upper_bounds[graph1_1]
        ax.axvline(x=-theta_constraint, color='k', lw=2, label='Limit')
        ax.axvline(x=theta_constraint, color='k', lw=2)
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_box_aspect(0.5)
        ax.legend(loc='upper right')

        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
            _, ax2 = plt.subplots()
            ax2.plot(np.linspace(0, 20, cert_results['obs'][0].shape[0])[1:], safety_filter.env.X_GOAL[:,0],'g--', label='Reference')
            ax2.plot(np.linspace(0, 20, uncert_results['obs'][0].shape[0]), uncert_results['obs'][0][:,0],'r--', label='Uncertified')
            ax2.plot(np.linspace(0, 20, cert_results['obs'][0].shape[0]), cert_results['obs'][0][:,0],'.-', label='Certified')
            ax2.plot(np.linspace(0, 20, cert_results['obs'][0].shape[0])[corrections], cert_results['obs'][0][corrections, 0], 'r.', label='Modified')
            ax2.set_xlabel(r'Time')
            ax2.set_ylabel(r'X')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')
        elif config.task == Environment.QUADROTOR:
            _, ax2 = plt.subplots()
            ax2.plot(uncert_results['obs'][0][:,graph3_1+1], uncert_results['obs'][0][:,graph3_2+1],'r--', label='Uncertified')
            ax2.plot(cert_results['obs'][0][:,graph3_1+1], cert_results['obs'][0][:,graph3_2+1],'.-', label='Certified')
            ax2.plot(cert_results['obs'][0][corrections, graph3_1+1], cert_results['obs'][0][corrections, graph3_2+1], 'r.', label='Modified')
            ax2.set_xlabel(r'x_dot')
            ax2.set_ylabel(r'z_dot')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')

        _, ax3 = plt.subplots()
        ax3.plot(uncert_results['obs'][0][:,graph3_1], uncert_results['obs'][0][:,graph3_2],'r--', label='Uncertified')
        ax3.plot(cert_results['obs'][0][:,graph3_1], cert_results['obs'][0][:,graph3_2],'.-', label='Certified')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(safety_filter.env.X_GOAL[:,graph3_1], safety_filter.env.X_GOAL[:,graph3_2],'g--', label='Reference')
        ax3.plot(cert_results['obs'][0][corrections, graph3_1], cert_results['obs'][0][corrections, graph3_2], 'r.', label='Modified')
        ax3.scatter(uncert_results['obs'][0][0, graph3_1], uncert_results['obs'][0][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        _, ax_act = plt.subplots()
        if config.task == Environment.CARTPOLE:
            ax_act.plot(cert_results['current_physical_action'][0][:], 'b-', label='Certified Input')
            ax_act.plot(mpsc_results['uncertified_action'][0][:], 'r--', label='Attempted Input')
            ax_act.plot(uncert_results['current_physical_action'][0][:], 'g--', label='Uncertified Input')
        else:
            ax_act.plot(cert_results['current_physical_action'][0][:, 0], 'b-', label='Certified Input 1')
            ax_act.plot(cert_results['current_physical_action'][0][:, 1], 'b--', label='Certified Input 2')
            ax_act.plot(mpsc_results['uncertified_action'][0][:, 0], 'r-', label='Attempted Input 1')
            ax_act.plot(mpsc_results['uncertified_action'][0][:, 1], 'r--', label='Attempted Input 2')
            ax_act.plot(uncert_results['current_physical_action'][0][:, 0], 'g-', label='Uncertified Input 1')
            ax_act.plot(uncert_results['current_physical_action'][0][:, 1], 'g--', label='Uncertified Input 2')
        ax_act.legend()
        ax_act.set_title('Input comparison')
        ax_act.set_xlabel('Step')
        ax_act.set_ylabel('Input')
        ax_act.set_box_aspect(0.5)

        _, ax_fft = plt.subplots()
        N_cert = max(cert_results['current_physical_action'][0].shape)
        N_uncert = max(uncert_results['current_physical_action'][0].shape)
        if config.task == Environment.CARTPOLE:
            spectrum_cert = fft(np.squeeze(cert_results['current_physical_action'][0]))
            spectrum_uncert = fft(np.squeeze(uncert_results['current_physical_action'][0]))
            freq_cert = fftfreq(len(spectrum_cert), 1/config.task_config['ctrl_freq'])[:N_cert//2]
            freq_uncert = fftfreq(len(spectrum_uncert), 1/config.task_config['ctrl_freq'])[:N_uncert//2]
            ax_fft.plot(freq_cert, 2.0/N_cert * np.abs(spectrum_cert[0:N_cert//2]), 'b-', label='Certified')
            ax_fft.plot(freq_uncert, 2.0/N_uncert * np.abs(spectrum_uncert[0:N_uncert//2]), 'r--', label='Uncertified')
            HFC_uncert = freq_uncert.T @ (2.0/N_uncert * np.abs(spectrum_uncert[0:N_uncert//2]))
            HFC_cert = freq_cert.T @ (2.0/N_cert * np.abs(spectrum_cert[0:N_cert//2]))
        else:
            spectrum_cert1 = fft(np.squeeze(cert_results['current_physical_action'][0][:, 0]))
            spectrum_cert2 = fft(np.squeeze(cert_results['current_physical_action'][0][:, 1]))
            spectrum_uncert1 = fft(np.squeeze(uncert_results['current_physical_action'][0][:, 0]))
            spectrum_uncert2 = fft(np.squeeze(uncert_results['current_physical_action'][0][:, 1]))
            freq_cert1 = fftfreq(len(spectrum_cert1), 1/config.task_config['ctrl_freq'])[:N_cert//2]
            freq_cert2 = fftfreq(len(spectrum_cert2), 1/config.task_config['ctrl_freq'])[:N_cert//2]
            freq_uncert1 = fftfreq(len(spectrum_uncert1), 1/config.task_config['ctrl_freq'])[:N_uncert//2]
            freq_uncert2 = fftfreq(len(spectrum_uncert2), 1/config.task_config['ctrl_freq'])[:N_uncert//2]
            ax_fft.plot(freq_cert1, 2.0/N_cert * np.abs(spectrum_cert1[0:N_cert//2]), 'b-', label='Certified 1')
            ax_fft.plot(freq_cert2, 2.0/N_cert * np.abs(spectrum_cert2[0:N_cert//2]), 'b-', label='Certified 2')
            ax_fft.plot(freq_uncert1, 2.0/N_uncert * np.abs(spectrum_uncert1[0:N_uncert//2]), 'r--', label='Uncertified 1')
            ax_fft.plot(freq_uncert2, 2.0/N_uncert * np.abs(spectrum_uncert2[0:N_uncert//2]), 'r--', label='Uncertified 2')
            HFC_uncert = freq_uncert1.T @ (2.0/N_uncert * np.abs(spectrum_uncert1[0:N_uncert//2])) + freq_uncert2.T @ (2.0/N_uncert * np.abs(spectrum_uncert2[0:N_uncert//2]))
            HFC_cert = freq_cert1.T @ (2.0/N_cert * np.abs(spectrum_cert1[0:N_cert//2])) + freq_cert2.T @ (2.0/N_cert * np.abs(spectrum_cert2[0:N_cert//2]))

        ax_fft.legend()
        ax_fft.set_title('Fourier Analysis of Inputs')
        ax_fft.set_xlabel('Frequency')
        ax_fft.set_ylabel('Magnitude')

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
        print('HFC Uncertified:', HFC_uncert)
        print('HFC Certified:', HFC_cert)
        print('RMSE Uncertified:', uncert_metrics['average_rmse'])
        print('RMSE Certified:', cert_metrics['average_rmse'])

        plt.tight_layout()
        plt.show()

    return uncert_results, uncert_metrics, cert_results, cert_metrics


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
    generator_env = env_func(init_state=None, randomized_init=True, init_state_randomization_info=reachable_state_randomization[system])

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

    if config.sf_config.cost_function == Cost_Function.LQR_COST:
        if config.algo == 'lqr':
            q_lin = config.algo_config.q_lqr
            r_lin = config.algo_config.r_lqr
        else:
            q_lin = [1]*safety_filter.model.nx
            r_lin = [0.1]
        safety_filter.cost_function.set_lqr_matrices(q_lin, r_lin)
    elif config.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl
        safety_filter.cost_function.output_dir = '.'
        if config.algo == 'pid':
            ctrl.save('./temp-data/saved_controller_prev.npy')
    elif config.sf_config.cost_function == Cost_Function.LEARNED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl
        if config.algo in ['ppo', 'sac']:
            safety_filter.cost_function.regularization_const = 200.0
        safety_filter.cost_function.learn_policy(path=f'./models/trajectories/{config.algo}_data_{system}_{task}.pkl')
        safety_filter.setup_optimizer()

    starting_points = []

    while len(starting_points) < num_points:
        generator_env.reset()
        init_state = generator_env.state
        test_env = env_func(init_state=init_state, randomized_init=False)

        uncert_experiment = Experiment(test_env, ctrl)
        cert_experiment = Experiment(test_env, ctrl, safety_filter=safety_filter)

        _, uncert_metrics = uncert_experiment.run_evaluation(n_episodes=1)
        uncert_experiment.reset()
        cert_results, cert_metrics = cert_experiment.run_evaluation(n_steps=2)
        cert_experiment.reset()
        test_env.close()

        mpsc_results = cert_results['safety_filter_data'][0]

        if np.all(mpsc_results['feasible']) \
                and uncert_metrics['average_constraint_violation'] > 5 \
                and uncert_metrics['average_length'] ==  config.task_config.ctrl_freq * config.task_config.episode_len_sec \
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
        uncert_results, _, cert_results, _ = run(plot=plot, training=False, n_episodes=1, n_steps=None, curr_path='.', init_state=init_state)
        if i == 0:
            all_uncert_results, all_cert_results = uncert_results, cert_results
        else:
            for key in all_cert_results.keys():
                if key in all_uncert_results:
                    all_uncert_results[key].append(uncert_results[key][0])
                all_cert_results[key].append(cert_results[key][0])

    uncert_metrics = Experiment.compute_metrics(all_uncert_results)
    cert_metrics = Experiment.compute_metrics(all_cert_results)

    all_results = {'uncert_results': all_uncert_results,
                   'uncert_metrics': uncert_metrics,
                   'cert_results': all_cert_results,
                   'cert_metrics': cert_metrics}

    with open(f'./results/{system}/{task}/m{config.sf_config.mpsc_cost_horizon}/results_{system}_{task}_{config.algo}_{config.sf_config.cost_function}_m{config.sf_config.mpsc_cost_horizon}.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    return all_uncert_results, uncert_metrics, all_cert_results, cert_metrics


if __name__ == '__main__':
    run()
    # determine_feasible_starting_points(num_points=10)
    # run_multiple(plot=False)
