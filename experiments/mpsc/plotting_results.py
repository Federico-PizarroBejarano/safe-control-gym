'''This script analyzes and plots the results from MPSC experiments.'''

import os
import pickle
from inspect import signature
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from safe_control_gym.experiments.base_experiment import MetricExtractor
from safe_control_gym.envs.benchmark_env import Task, Environment
from safe_control_gym.safety_filters.mpsc.mpsc_utils import high_frequency_content, get_discrete_derivative


plot = False
save_figs = True
ordered_algos = ['lqr', 'ppo', 'sac']
# ordered_algos = ['lqr', 'pid', 'ppo', 'sac']

# cost_colors = {'one_step':'cornflowerblue', 'constant': 'goldenrod', 'regularized': 'pink', 'lqr':'tomato', 'precomputed':'limegreen', 'learned':'yellow'}
cost_colors = {'one_step': 'cornflowerblue', 'regularized': 'pink', 'precomputed M=2': 'limegreen', 'precomputed M=5': 'forestgreen', 'precomputed M=10': 'darkgreen'}

U_EQs = {
    'cartpole': 0,
    'quadrotor_2D': 0.1323,
    'quadrotor_3D': 0.06615
}

met = MetricExtractor()
met.verbose = False


def load_one_experiment(system, task, algo, mpsc_cost_horizon):
    '''Loads the results of every MPSC cost function for a specific experiment.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
        algo (str): The controller being used.
        mpsc_cost_horizon (str): The cost horizon used by the smooth MPSC cost functions.

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    all_results = {}

    for cost in ordered_costs:
        with open(f'./results_mpsc/{system}/{task}/m{mpsc_cost_horizon}/results_{system}_{task}_{algo}_{cost}_cost_m{mpsc_cost_horizon}.pkl', 'rb') as f:
            all_results[cost] = pickle.load(f)

    return all_results


def load_all_algos(system, task, mpsc_cost_horizon):
    '''Loads the results of every MPSC cost function for a specific experiment with every algo.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
        mpsc_cost_horizon (str): The cost horizon used by the smooth MPSC cost functions.

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    all_results = {}

    for algo in ['lqr', 'pid', 'ppo', 'sac']:
        if system == 'cartpole' and algo == 'pid':
            continue

        all_results[algo] = load_one_experiment(system, task, algo, mpsc_cost_horizon)

    return all_results


def load_all_models(system, task, algo):
    '''Loads the results of every MPSC cost function for a specific experiment with every algo.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    all_results = {}

    for model in os.listdir(f'./models/rl_models/{system}/{task}/{algo}/'):
        with open(f'./results_mpsc/{system}/{task}/results_{system}_{task}_{algo}_{model}.pkl', 'rb') as f:
            all_results[model] = pickle.load(f)

    with open(f'./results_mpsc/{system}/{task}/results_{system}_{task}_safe_explorer_ppo_none.pkl', 'rb') as f:
            all_results['safe_ppo'] = pickle.load(f)
            all_results['safe_ppo_cert'] = all_results['safe_ppo']

    return all_results


def plot_experiment(system, task, mpsc_cost_horizon, data_extractor):
    '''Plots the results of every MPSC cost function for a specific experiment.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
        mpsc_cost_horizon (str): The cost horizon used by the smooth MPSC cost functions.
        data_extractor (lambda): A function that extracts the necessary data from the results.
    '''

    all_results = load_all_algos(system, task, mpsc_cost_horizon)

    if len(signature(data_extractor).parameters) > 1:
        show_uncertified = True
    else:
        show_uncertified = False

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    uncertified_data = []
    cert_data = defaultdict(list)
    labels = []

    for algo in ordered_algos:
        if algo not in all_results:
            continue
        labels.append(algo.upper())

        for cost in ordered_costs:
            raw_data = all_results[algo][cost]
            cert_data[cost].append(data_extractor(raw_data))
            if show_uncertified and cost == 'one_step':
                uncertified_data.append(data_extractor(raw_data, certified=False))

    num_bars = len(ordered_costs) + show_uncertified + 1
    width = 1 / (num_bars + 1)
    widths = [width] * len(labels)
    x = np.arange(1, len(labels) + 1)

    box_plots = {}
    medianprops = dict(linestyle='--', linewidth=2.5, color='black')

    cost_names = []
    if show_uncertified:
        cost_names.append('Uncertified')
        box_plots['uncertified'] = ax.boxplot(uncertified_data, positions=x - (num_bars - 1) / 2.0 * width, widths=widths, patch_artist=True, medianprops=medianprops)
        for patch in box_plots['uncertified']['boxes']:
            patch.set_facecolor('plum')

    for idx, cost in enumerate(ordered_costs):
        cost_name = cost.replace('_', ' ').title()
        if cost_name == 'Lqr':
            cost_name = 'LQR'
        cost_names.append(f'{cost_name} Cost')
        position = ((num_bars - 1) / 2.0 - idx - show_uncertified) * width
        box_plots[cost] = ax.boxplot(cert_data[cost], positions=x - position, widths=widths, patch_artist=True, medianprops=medianprops)

        for patch in box_plots[cost]['boxes']:
            patch.set_facecolor(cost_colors[cost])

    ylabel = data_extractor.__name__.replace('extract_', '').replace('_', ' ').title()
    if ylabel == 'Rmse':
        ylabel = 'RMSE'
    ax.set_ylabel(ylabel, weight='bold', fontsize=25, labelpad=10)
    task_title = 'Stabilization' if task == 'stab' else 'Trajectory Tracking'
    ax.set_title(f'{system.title()} {task_title} {ylabel} with M={mpsc_cost_horizon}', weight='bold', fontsize=25)

    ax.set_xticks(x, labels, weight='bold', fontsize=25)
    first_boxes = [box_plots[cost]['boxes'][0] for cost in ordered_costs]
    if show_uncertified:
        first_boxes = [box_plots['uncertified']['boxes'][0]] + first_boxes
    ax.legend(first_boxes, cost_names, loc='upper right', fontsize=25)

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)
    if plot is True:
        plt.show()

    image_suffix = data_extractor.__name__.replace('extract_', '')
    if save_figs:
        fig.savefig(f'./results_mpsc/{system}/{task}/m{mpsc_cost_horizon}/graphs/{system}_{task}_{image_suffix}_m{mpsc_cost_horizon}.png', dpi=300)


def plot_violations(system, task, mpsc_cost_horizon):
    '''Plots the constraint violations of every controller for a specific experiment.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
        mpsc_cost_horizon (str): The cost horizon used by the smooth MPSC cost functions.
    '''

    all_results = load_all_algos(system, task, mpsc_cost_horizon)

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = []
    data = []

    for algo in ordered_algos:
        if algo not in all_results:
            continue
        labels.append(algo.upper())

        one_step_cost = all_results[algo]['one_step']
        data.append(extract_constraint_violations(one_step_cost, certified=False))

    ax.set_ylabel('Number of Constraint Violations', weight='bold', fontsize=25, labelpad=10)
    task_title = 'Stabilization' if task == 'stab' else 'Trajectory Tracking'
    ax.set_title(f'{system.title()} {task_title} Constraint Violations with M={mpsc_cost_horizon}', weight='bold', fontsize=25)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x, labels, weight='bold', fontsize=25)

    medianprops = dict(linestyle='--', linewidth=2.5, color='black')
    bplot = ax.boxplot(data, patch_artist=True, labels=labels, medianprops=medianprops, widths=[0.75] * len(labels))

    cm = plt.cm.get_cmap('inferno', len(labels) + 2)
    colors = [cm(i) for i in range(1, len(labels) + 1)]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    if save_figs:
        fig.savefig(f'./results_mpsc/{system}/{task}/m{mpsc_cost_horizon}/graphs/{system}_{task}_constraint_violations.png', dpi=300)


def extract_magnitude_of_corrections(results_data):
    '''Extracts the magnitude of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        magn_of_corrections (list): The list of magnitude of corrections for all experiments.
    '''

    magn_of_corrections = [np.linalg.norm(mpsc_results['correction'][0]) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    return magn_of_corrections


def extract_max_correction(results_data):
    '''Extracts the max correction from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        max_corrections (list): The list of max corrections for all experiments.
    '''
    max_corrections = [np.max(np.abs(mpsc_results['correction'][0])) for mpsc_results in results_data['cert_results']['safety_filter_data']]

    return max_corrections


def extract_number_of_corrections(results_data):
    '''Extracts the number of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        num_corrections (list): The list of the number of corrections for all experiments.
    '''
    num_corrections = [np.sum(mpsc_results['correction'][0] * 10.0 > np.linalg.norm(results_data['cert_results']['current_clipped_action'][i] - U_EQs[system_name], axis=1)) for i, mpsc_results in enumerate(results_data['cert_results']['safety_filter_data'])]
    return num_corrections


def extract_rmse(results_data, certified=True):
    '''Extracts the RMSEs from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        rmse (list): The list of RMSEs for all experiments.
    '''
    if certified:
        met.data = results_data['cert_results']
        rmse = np.asarray(met.get_episode_rmse())
    else:
        met.data = results_data['uncert_results']
        rmse = np.asarray(met.get_episode_rmse())
    return rmse


def extract_length(results_data, certified=True):
    '''Extracts the lengths from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        length (list): The list of lengths for all experiments.
    '''
    if certified:
        met.data = results_data['cert_results']
        length = np.asarray(met.get_episode_lengths())
    else:
        met.data = results_data['uncert_results']
        length = np.asarray(met.get_episode_lengths())
    return length


def extract_simulation_time(results_data, certified=True):
    '''Extracts the simulation time from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        sim_time (list): The list of simulation times for all experiments.
    '''
    if certified:
        sim_time = [timestamp[-1] - timestamp[0] for timestamp in results_data['cert_results']['timestamp']]
    else:
        sim_time = [timestamp[-1] - timestamp[0] for timestamp in results_data['uncert_results']['timestamp']]

    return sim_time


def extract_constraint_violations(results_data, certified=True):
    '''Extracts the simulation time from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        num_violations (list): The list of number of constraint violations for all experiments.
    '''
    if certified:
        met.data = results_data['cert_results']
        num_violations = np.asarray(met.get_episode_constraint_violation_steps())
    else:
        met.data = results_data['uncert_results']
        num_violations = np.asarray(met.get_episode_constraint_violation_steps())

    return num_violations


def extract_high_frequency_content(results_data, certified=True):
    '''Extracts the high frequency content (HFC) from the inputs of an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        HFC (list): The list of HFCs for all experiments.
    '''
    n = min(results_data['cert_results']['current_clipped_action'][0].shape)

    if certified:
        all_actions = results_data['cert_results']['current_clipped_action']
    else:
        all_actions = results_data['uncert_results']['current_clipped_action']

    HFC = []
    for actions in all_actions:
        if n == 1:
            ctrl_freq = 15
        elif n > 1:
            ctrl_freq = 50
        HFC.append(high_frequency_content(actions - U_EQs[system_name], ctrl_freq))

    return np.squeeze(HFC)


def extract_rate_of_change(results_data, certified=True, order=1, mode='input'):
    '''Extracts the rate of change of a signal from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.
        order (int): Either 1 or 2, denoting the order of the derivative.
        mode (string): Either 'input' or 'correction', denoting which signal to use.

    Returns:
        roc (list): The list of rate of changes.
    '''
    n = min(results_data['cert_results']['current_clipped_action'][0].shape)

    if mode == 'input':
        if certified:
            all_signals = [actions - U_EQs[system_name] for actions in results_data['cert_results']['current_clipped_action']]
        else:
            all_signals = [actions - U_EQs[system_name] for actions in results_data['uncert_results']['current_clipped_action']]
    elif mode == 'correction':
        all_signals = [np.squeeze(mpsc_results['uncertified_action'][0]) - np.squeeze(mpsc_results['certified_action'][0]) for mpsc_results in results_data['cert_results']['safety_filter_data']]

    total_derivatives = []
    for signal in all_signals:
        if n == 1:
            ctrl_freq = 15
            if mode == 'correction':
                signal = np.atleast_2d(signal).T
        elif n > 1:
            ctrl_freq = 50
        derivative = get_discrete_derivative(signal, ctrl_freq)
        if order == 2:
            derivative = get_discrete_derivative(derivative, ctrl_freq)
        total_derivatives.append(np.linalg.norm(derivative, 'fro'))

    return total_derivatives


def extract_number_of_correction_intervals(results_data):
    '''Extracts the frequency the safety filter turns on or off from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        num_correction_intervals (list): The list of number of times the filter starts correcting.
    '''
    all_corrections = [(mpsc_results['correction'][0] * 10.0 > np.linalg.norm(results_data['cert_results']['current_clipped_action'][i] - U_EQs[system_name], axis=1)) for i, mpsc_results in enumerate(results_data['cert_results']['safety_filter_data'])]

    correction_frequency = []
    for corrections in all_corrections:
        correction_frequency.append((np.diff(corrections) != 0).sum())

    return correction_frequency


def extract_reward(results_data, certified):
    '''Extracts the mean reward from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        mean_reward (list): The list of mean rewards.
    '''
    if certified:
        met.data = results_data['cert_results']
        returns = np.asarray(met.get_episode_returns())
    else:
        met.data = results_data['uncert_results']
        returns = np.asarray(met.get_episode_returns())

    return returns


def extract_final_dist(results_data, certified):
    '''Extracts the final distance from stabilization goal from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        final_dist (list): The list of final distances.
    '''
    if certified:
        data = results_data['cert_results']
    else:
        data = results_data['uncert_results']

    if results_data['X_GOAL'].ndim < 2:
        final_dists = [np.linalg.norm(results_data['X_GOAL'] - data['state'][i][-1]) for i in range(len(data['obs']))]
    else:
        final_dists = [np.linalg.norm(results_data['X_GOAL'][:len(data['state'][i][:, 0]), 0] - data['state'][i][:, 0]) for i in range(len(data['obs']))]

    return final_dists


def extract_failed(results_data, certified):
    '''Extracts the percent failed from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        failed (list): The percent failed.
    '''
    if certified:
        data = results_data['cert_results']
    else:
        data = results_data['uncert_results']

    failed = [data['info'][i][-1]['out_of_bounds'] for i in range(len(data['info']))]

    return [np.mean(failed)]


def plot_trajectories(config, X_GOAL, uncert_results, cert_results):
    '''Plots a series of graphs detailing the experiments in the passed in data.

    Args:
        config (dict): The configuration of the experiment.
        X_GOAL (np.ndarray): The goal (stabilization or reference trajectory) of the experiment.
        uncert_results (dict): The results of the uncertified experiment.
        cert_results (dict): The results of the certified experiment.
    '''
    met.data = cert_results
    print('Total Certified Violations:', np.asarray(met.get_episode_constraint_violation_steps()).sum())

    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    for exp in range(len(uncert_results['obs'])):
        specific_results = {key: [cert_results[key][exp]] for key in cert_results.keys()}
        met.data = specific_results
        print(f'Total Certified Violations ({exp}):', np.asarray(met.get_episode_constraint_violation_steps()).sum())
        mpsc_results = cert_results['safety_filter_data'][exp]
        corrections = mpsc_results['correction'][0] * 10.0 > np.linalg.norm(cert_results['current_clipped_action'][exp] - U_EQs[system], axis=1)
        corrections = np.append(corrections, False)

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
        ax.plot(uncert_results['obs'][exp][:, graph1_1], uncert_results['obs'][exp][:, graph1_2], 'r--', label='Uncertified')
        ax.plot(cert_results['obs'][exp][:, graph1_1], cert_results['obs'][exp][:, graph1_2], '.-', label='Certified')
        ax.plot(cert_results['obs'][exp][corrections, graph1_1], cert_results['obs'][exp][corrections, graph1_2], 'r.', label='Modified')
        ax.scatter(uncert_results['obs'][exp][0, graph1_1], uncert_results['obs'][exp][0, graph1_2], color='g', marker='o', s=100, label='Initial State')
        theta_constraint = config.task_config['constraints'][0].upper_bounds[graph1_1]
        ax.axvline(x=-theta_constraint, color='k', lw=2, label='Limit')
        ax.axvline(x=theta_constraint, color='k', lw=2)
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$\dot{\theta}$')
        ax.set_box_aspect(0.5)
        ax.legend(loc='upper right')

        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.CARTPOLE:
            _, ax2 = plt.subplots()
            ax2.plot(np.linspace(0, 20, cert_results['obs'][exp].shape[0]), X_GOAL[:, 0], 'g--', label='Reference')
            ax2.plot(np.linspace(0, 20, uncert_results['obs'][exp].shape[0]), uncert_results['obs'][exp][:, 0], 'r--', label='Uncertified')
            ax2.plot(np.linspace(0, 20, cert_results['obs'][exp].shape[0]), cert_results['obs'][exp][:, 0], '.-', label='Certified')
            ax2.plot(np.linspace(0, 20, cert_results['obs'][exp].shape[0])[corrections], cert_results['obs'][exp][corrections, 0], 'r.', label='Modified')
            ax2.set_xlabel(r'Time')
            ax2.set_ylabel(r'X')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')
        elif config.task == Environment.QUADROTOR:
            _, ax2 = plt.subplots()
            ax2.plot(uncert_results['obs'][exp][:, graph3_1 + 1], uncert_results['obs'][exp][:, graph3_2 + 1], 'r--', label='Uncertified')
            ax2.plot(cert_results['obs'][exp][:, graph3_1 + 1], cert_results['obs'][exp][:, graph3_2 + 1], '.-', label='Certified')
            ax2.plot(cert_results['obs'][exp][corrections, graph3_1 + 1], cert_results['obs'][exp][corrections, graph3_2 + 1], 'r.', label='Modified')
            ax2.set_xlabel(r'x_dot')
            ax2.set_ylabel(r'z_dot')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')

        _, ax3 = plt.subplots()
        ax3.plot(uncert_results['obs'][exp][:, graph3_1], uncert_results['obs'][exp][:, graph3_2], 'r--', label='Uncertified')
        ax3.plot(cert_results['obs'][exp][:, graph3_1], cert_results['obs'][exp][:, graph3_2], '.-', label='Certified')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(X_GOAL[:, graph3_1], X_GOAL[:, graph3_2], 'g--', label='Reference')
        ax3.plot(cert_results['obs'][exp][corrections, graph3_1], cert_results['obs'][exp][corrections, graph3_2], 'r.', label='Modified')
        ax3.scatter(uncert_results['obs'][exp][0, graph3_1], uncert_results['obs'][exp][0, graph3_2], color='g', marker='o', s=100, label='Initial State')
        ax3.set_xlabel(r'X')
        if config.task == Environment.CARTPOLE:
            ax3.set_ylabel(r'Vel')
        elif config.task == Environment.QUADROTOR:
            ax3.set_ylabel(r'Z')
        ax3.set_box_aspect(0.5)
        ax3.legend(loc='upper right')

        _, ax_act = plt.subplots()
        if config.task == Environment.CARTPOLE:
            ax_act.plot(cert_results['current_clipped_action'][exp][:], 'b-', label='Certified Input')
            ax_act.plot(mpsc_results['uncertified_action'][0][:], 'r--', label='Attempted Input')
            ax_act.plot(uncert_results['current_clipped_action'][exp][:], 'g--', label='Uncertified Input')
        else:
            ax_act.plot(cert_results['current_clipped_action'][exp][:, 0], 'b-', label='Certified Input 1')
            ax_act.plot(mpsc_results['uncertified_action'][0][:, 0], 'r-', label='Attempted Input 1')
            ax_act.plot(uncert_results['current_clipped_action'][exp][:, 0], 'g-', label='Uncertified Input 1')
        ax_act.legend()
        ax_act.set_title('Input comparison')
        ax_act.set_xlabel('Step')
        ax_act.set_ylabel('Input')
        ax_act.set_box_aspect(0.5)

        plt.tight_layout()
        plt.show()


def create_paper_plot(system, task, data_extractor):
    '''Plots the results of every paper MPSC cost function for a specific experiment.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
        mpsc_cost_horizon (str): The cost horizon used by the smooth MPSC cost functions.
        data_extractor (lambda): A function that extracts the necessary data from the results.
    '''

    if len(signature(data_extractor).parameters) > 1:
        show_uncertified = True
    else:
        show_uncertified = False

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    uncertified_data = []
    cert_data = defaultdict(list)

    labels = [algo.upper() for algo in ordered_algos]

    ordered_cost_ids = ['one_step', 'regularized', 'precomputed M=2', 'precomputed M=5', 'precomputed M=10']
    for cost_name in ordered_cost_ids:
        if cost_name in ['one_step', 'regularized']:
            all_results = load_all_algos(system, task, 10)
            cost = cost_name
        else:
            M = int(cost_name.split('=')[1])
            all_results = load_all_algos(system, task, M)
            cost = 'precomputed'

        for algo in ordered_algos:
            raw_data = all_results[algo][cost]
            cert_data[cost_name].append(data_extractor(raw_data))
            if show_uncertified and cost == 'one_step':
                uncertified_data.append(data_extractor(raw_data, certified=False))

    num_bars = len(ordered_cost_ids) + show_uncertified
    width = 1 / (num_bars + 1)
    widths = [width] * len(labels)
    x = np.arange(1, len(labels) + 1)

    box_plots = {}
    medianprops = dict(linestyle='--', linewidth=2.5, color='black')
    flierprops = {'marker': 'o', 'markersize': 3}

    cost_names = []
    if show_uncertified:
        cost_names.append('Uncertified')
        box_plots['uncertified'] = ax.boxplot(uncertified_data, positions=x - (num_bars - 1) / 2.0 * width, widths=widths, patch_artist=True, medianprops=medianprops, flierprops=flierprops)
        for patch in box_plots['uncertified']['boxes']:
            patch.set_facecolor('plum')

    for idx, cost_id in enumerate(ordered_cost_ids):
        cost = cost_id.split(' ')[0]
        cost_name = cost_id.title()
        if cost_name == 'Lqr':
            cost_name = 'LQR'
        cost_names.append(f'{cost_name} Cost')
        position = ((num_bars - 1) / 2.0 - idx - show_uncertified) * width
        box_plots[cost_id] = ax.boxplot(cert_data[cost_id], positions=x - position, widths=widths, patch_artist=True, medianprops=medianprops)

        for patch in box_plots[cost_id]['boxes']:
            patch.set_facecolor(cost_colors[cost_id])

    ylabel = data_extractor.__name__.replace('extract_', '').replace('_', ' ').title()
    if ylabel == 'Rmse':
        ylabel = 'RMSE'
    ax.set_ylabel(ylabel, weight='bold', fontsize=45, labelpad=10)

    ax.set_xticks(x, labels, weight='bold', fontsize=45)
    ax.set_xlim([0.5, len(labels) + 0.5])

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)
    if plot is True:
        plt.show()

    image_suffix = data_extractor.__name__.replace('extract_', '')
    if save_figs:
        fig.savefig(f'./results_mpsc/{system}_{task}_{image_suffix}.png', dpi=300)


def plot_model_comparisons(system, task, algo, data_extractor):
    '''Plots the constraint violations of every controller for a specific experiment.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
        mpsc_cost_horizon (str): The cost horizon used by the smooth MPSC cost functions.
    '''

    all_results = load_all_models(system, task, algo)

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = sorted(os.listdir(f'./models/rl_models/{system}/{task}/{algo}/'))
    if 'cert' not in data_extractor.__name__:
        labels = sorted(labels + ['safe_ppo_cert'])
    elif 'uncert' in data_extractor.__name__:
        labels = sorted(labels + ['safe_ppo'])
    else:
        labels = sorted(labels + ['safe_ppo'] + ['safe_ppo_cert'])

    data = []

    for model in labels:
        exp_data = all_results[model]
        if model == 'safe_ppo':
            data.append(data_extractor(exp_data, certified=False))
        else:
            data.append(data_extractor(exp_data))

    ylabel = data_extractor.__name__.replace('extract_', '').replace('_', ' ').title()
    ax.set_ylabel(ylabel, weight='bold', fontsize=45, labelpad=10)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x, labels, weight='bold', fontsize=15, rotation=45, ha='right')

    medianprops = dict(linestyle='--', linewidth=2.5, color='black')
    bplot = ax.boxplot(data, patch_artist=True, labels=labels, medianprops=medianprops, widths=[0.75] * len(labels))

    cm = mpl.colormaps['inferno']
    colors = [cm(i/len(labels)) for i in range(1, len(labels)+1)]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    fig.tight_layout()

    if data_extractor != extract_reward_cert:
        ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    if save_figs:
        image_suffix = data_extractor.__name__.replace('extract_', '')
        fig.savefig(f'./results_mpsc/{system}/{task}/graphs/{algo}/{system}_{task}_{image_suffix}.png', dpi=300)


if __name__ == '__main__':
    ordered_costs = ['one_step', 'regularized', 'precomputed']

    def extract_rate_of_change_of_inputs(results_data, certified=True): return extract_rate_of_change(results_data, certified, order=1, mode='input')

    def extract_roc_cert(results_data, certified=True): return extract_rate_of_change_of_inputs(results_data, certified)
    def extract_roc_uncert(results_data, certified=False): return extract_rate_of_change_of_inputs(results_data, certified)

    def extract_rmse_cert(results_data, certified=True): return extract_rmse(results_data, certified)
    def extract_rmse_uncert(results_data, certified=False): return extract_rmse(results_data, certified)

    def extract_constraint_violations_cert(results_data, certified=True): return extract_constraint_violations(results_data, certified)
    def extract_constraint_violations_uncert(results_data, certified=False): return extract_constraint_violations(results_data, certified)

    def extract_reward_cert(results_data, certified=True): return extract_reward(results_data, certified)
    def extract_reward_uncert(results_data, certified=False): return extract_reward(results_data, certified)

    def extract_final_dist_cert(results_data, certified=True): return extract_final_dist(results_data, certified)
    def extract_final_dist_uncert(results_data, certified=False): return extract_final_dist(results_data, certified)

    def extract_failed_cert(results_data, certified=True): return extract_failed(results_data, certified)
    def extract_failed_uncert(results_data, certified=False): return extract_failed(results_data, certified)

    def extract_length_cert(results_data, certified=True): return extract_length(results_data, certified)
    def extract_length_uncert(results_data, certified=False): return extract_length(results_data, certified)

    system_name = 'cartpole'
    task_name = 'stab'
    algo_name = 'ppo'
    plot_model_comparisons(system_name, task_name, algo_name, extract_magnitude_of_corrections)
    plot_model_comparisons(system_name, task_name, algo_name, extract_max_correction)
    plot_model_comparisons(system_name, task_name, algo_name, extract_roc_cert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_roc_uncert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_rmse_cert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_rmse_uncert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_constraint_violations_cert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_constraint_violations_uncert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_number_of_corrections)
    plot_model_comparisons(system_name, task_name, algo_name, extract_length_cert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_length_uncert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_reward_cert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_reward_uncert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_failed_cert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_failed_uncert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_final_dist_cert)
    plot_model_comparisons(system_name, task_name, algo_name, extract_final_dist_uncert)

    # mpsc_cost_horizon_num = 2

    # dataframe = generate_dataframe()
    # create_table(dataframe, 'cartpole')
    # create_table(dataframe, 'quadrotor_2D')
    # create_table(dataframe, 'quadrotor_3D')

    # for system_name in ['cartpole', 'quadrotor_2D', 'quadrotor_3D']:
    #     for task_name in ['stab', 'track']:
    #         plot_violations(system_name, task_name, mpsc_cost_horizon_num)

    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_rate_of_change_of_inputs)
    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_rate_of_change_of_corrections)

    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_high_frequency_content)

    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_magnitude_of_corrections)
    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_max_correction)

    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_simulation_time)
    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_rmse)

    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_number_of_corrections)
    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_number_of_correction_intervals)

    #         plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_constraint_violations)

    # Plotting a single experiment
    # algo_name = 'lqr'
    # mpsc_cost_name = 'one_step'
    # one_result = load_one_experiment(system=system_name, task=task_name, algo=algo_name, mpsc_cost_horizon=mpsc_cost_horizon_num)
    # results = one_result[mpsc_cost_name]
    # plot_trajectories(results['config'], results['X_GOAL'], results['uncert_results'], results['cert_results'])
