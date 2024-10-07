'''This script analyzes and plots the results from MPSC experiments.'''

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.envs.benchmark_env import Environment, Task
from safe_control_gym.experiments.base_experiment import MetricExtractor
from safe_control_gym.safety_filters.mpsc.mpsc_utils import get_discrete_derivative, high_frequency_content
from safe_control_gym.utils.plotting import load_from_logs

plot = True  # Saves figure if False

U_EQs = {
    'cartpole': 0,
    'quadrotor_2D': 0.1323,
    'quadrotor_3D': 0.06615
}

met = MetricExtractor()
met.verbose = False


def load_all_models(system, task, algo):
    '''Loads the results of every experiment.

    Args:
        system (str): The system to be plotted.
        task (str): The task to be plotted (either 'stab' or 'track').
        algo (str): The controller to be plotted.

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    all_results = {}

    for model in ordered_models:
        all_results[model] = []
        for seed in os.listdir(f'./results_mpsc/{system}/{task}/{algo}/results_{system}_{task}_{algo}_{model}/'):
            with open(f'./results_mpsc/{system}/{task}/{algo}/results_{system}_{task}_{algo}_{model}/{seed}', 'rb') as f:
                all_results[model].append(pickle.load(f))
        consolidate_multiple_seeds(all_results, model)

    return all_results


def consolidate_multiple_seeds(all_results, model):
    all_data = all_results[model]
    data = all_results[model][0]
    all_results[model] = {}
    all_results[model]['X_GOAL'] = data['X_GOAL']
    all_results[model]['config'] = data['config']
    all_results[model]['cert_metrics'] = data['cert_metrics']
    all_results[model]['uncert_metrics'] = data['uncert_metrics']

    for k in data['cert_results'].keys():
        for i in range(1, len(all_data)):
            data['cert_results'][k] += all_data[i]['cert_results'][k]

    for k in data['uncert_results'].keys():
        for i in range(1, len(all_data)):
            data['uncert_results'][k] += all_data[i]['uncert_results'][k]

    all_results[model]['cert_results'] = data['cert_results']
    all_results[model]['uncert_results'] = data['uncert_results']


def extract_magnitude_of_corrections(results_data):
    '''Extracts the magnitude of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        magn_of_corrections (list): The list of magnitude of corrections for all experiments.
    '''

    magn_of_corrections = [np.linalg.norm(mpsc_results['correction'][0]) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    return magn_of_corrections


def extract_percent_magnitude_of_corrections(results_data):
    '''Extracts the percent magnitude of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        magn_of_corrections (list): The list of percent magnitude of corrections for all experiments.
    '''

    N = len(results_data['cert_results']['state'][0]) - 1
    norm_uncert = [normalize_actions(mpsc_results['uncertified_action'][0]).reshape((N, -1)) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    norm_cert = [normalize_actions(mpsc_results['certified_action'][0]).reshape((N, -1)) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    corr = [(norm_uncert[i] - norm_cert[i]) for i in range(len(norm_cert))]
    max_input = [np.maximum(np.linalg.norm(norm_uncert[i], axis=1), np.linalg.norm(norm_cert[i], axis=1)) for i in range(len(norm_cert))]
    perc_change = [np.divide(np.linalg.norm(corr[i], axis=1), max_input[i]) for i in range(len(norm_cert))]
    magn_of_corrections = [np.linalg.norm(elem) for elem in perc_change]

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


def extract_percent_max_correction(results_data):
    '''Extracts the percent max correction from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        max_corrections (list): The list of percent max corrections for all experiments.
    '''
    N = len(results_data['cert_results']['state'][0]) - 1
    norm_uncert = [normalize_actions(mpsc_results['uncertified_action'][0]).reshape((N, -1)) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    norm_cert = [normalize_actions(mpsc_results['certified_action'][0]).reshape((N, -1)) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    corr = [(norm_uncert[i] - norm_cert[i]) for i in range(len(norm_cert))]
    max_input = [np.maximum(np.linalg.norm(norm_uncert[i], axis=1), np.linalg.norm(norm_cert[i], axis=1)) for i in range(len(norm_cert))]
    perc_change = [np.divide(np.linalg.norm(corr[i], axis=1), max_input[i]) for i in range(len(norm_cert))]
    max_corrections = [np.max(elem) for elem in perc_change]

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


def extract_feasible_iterations(results_data):
    '''Extracts the number of feasible iterations from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        feasible_iterations (list): The list of the number of feasible iterations for all experiments.
    '''
    feasible_iterations = [np.sum(mpsc_results['feasible'][0]) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    return feasible_iterations


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


def plot_model_comparisons(system, task, algo, data_extractor):
    '''Plots the constraint violations of every controller for a specific experiment.

    Args:
        system (str): The system to be plotted.
        task (str): The task to be plotted (either 'stab' or 'track').
        algo (str): The controller to be plotted.
        data_extractor (func): The function which extracts the desired data.
    '''

    all_results = load_all_models(system, task, algo)

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = ordered_models

    data = []

    for model in ordered_models:
        exp_data = all_results[model]
        data.append(data_extractor(exp_data))

    ylabel = data_extractor.__name__.replace('extract_', '').replace('_', ' ').title()
    ax.set_ylabel(ylabel, weight='bold', fontsize=45, labelpad=10)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x, labels, weight='bold', fontsize=15, rotation=30, ha='right')

    medianprops = dict(linestyle='--', linewidth=2.5, color='black')
    bplot = ax.boxplot(data, patch_artist=True, labels=labels, medianprops=medianprops, widths=[0.75] * len(labels), showfliers=False)

    for patch, color in zip(bplot['boxes'], colors.values()):
        patch.set_facecolor(color)

    fig.tight_layout()

    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    else:
        image_suffix = data_extractor.__name__.replace('extract_', '')
        fig.savefig(f'./results_mpsc/{image_suffix}.png', dpi=300)
    plt.close()


def plot_step_time(system, task, algo):
    '''Plots the constraint violations of every controller for a specific experiment.

    Args:
        system (str): The system to be plotted.
        task (str): The task to be plotted (either 'stab' or 'track').
        algo (str): The controller to be plotted.
    '''

    all_results = {}
    for model in ordered_models:
        all_results[model] = []
        for seed in os.listdir(f'./models/rl_models/{system}/{task}/{algo}/{model}/'):
            all_results[model].append(load_from_logs(f'./models/rl_models/{system}/{task}/{algo}/{model}/{seed}/logs/'))

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = ordered_models

    data = []

    for model in ordered_models:
        datum = np.array([values['stat/step_time'][3] for values in all_results[model]]).flatten()
        data.append(datum)

    ylabel = 'Training Time per Step [ms]'
    ax.set_ylabel(ylabel, weight='bold', fontsize=45, labelpad=10)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks(x, labels, weight='bold', fontsize=15, rotation=30, ha='right')

    medianprops = dict(linestyle='--', linewidth=2.5, color='black')
    bplot = ax.boxplot(data, patch_artist=True, labels=labels, medianprops=medianprops, widths=[0.75] * len(labels), showfliers=False)

    for patch, color in zip(bplot['boxes'], colors.values()):
        patch.set_facecolor(color)

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    else:
        image_suffix = 'step_time'
        fig.savefig(f'./results_mpsc/{image_suffix}.png', dpi=300)
    plt.close()


def normalize_actions(actions):
    '''Normalizes an array of actions.

    Args:
        actions (ndarray): The actions to be normalized.

    Returns:
        normalized_actions (ndarray): The normalized actions.
    '''
    if system_name == 'cartpole':
        action_scale = 10.0
        normalized_actions = actions / action_scale
    elif system_name == 'quadrotor_2D':
        hover_thrust = 0.1323
        norm_act_scale = 0.1
        normalized_actions = (actions / hover_thrust - 1.0) / norm_act_scale
    else:
        hover_thrust = 0.06615
        norm_act_scale = 0.1
        normalized_actions = (actions / hover_thrust - 1.0) / norm_act_scale

    return normalized_actions


def plot_all_logs(system, task, algo):
    '''Plots comparative plots of all the logs.

    Args:
        system (str): The system to be plotted.
        task (str): The task to be plotted (either 'stab' or 'track').
        algo (str): The controller to be plotted.
    '''
    all_results = {}

    for model in ordered_models:
        all_results[model] = []
        for seed in os.listdir(f'./models/rl_models/{system}/{task}/{algo}/{model}/'):
            all_results[model].append(load_from_logs(f'./models/rl_models/{system}/{task}/{algo}/{model}/{seed}/logs/'))

    for key in all_results[ordered_models[0]][0].keys():
        if key == 'stat_eval/ep_return':
            plot_log(key, all_results)
        if key == 'stat/constraint_violation':
            plot_log(key, all_results)


def plot_log(key, all_results):
    '''Plots a comparative plot of the log 'key'.

    Args:
        key (str): The name of the log to be plotted.
        all_results (dict): A dictionary of all the logged results for all models.
    '''
    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = ordered_models

    for model, label in zip(ordered_models, labels):
        x = all_results[model][0][key][1] / 1000
        all_data = np.array([values[key][3] for values in all_results[model]])
        ax.plot(x, np.mean(all_data, axis=0), label=label, color=colors[model])
        ax.fill_between(x, np.min(all_data, axis=0), np.max(all_data, axis=0), alpha=0.3, edgecolor=colors[model], facecolor=colors[model])

    ax.set_ylabel(key, weight='bold', fontsize=45, labelpad=10)
    ax.set_xlabel('Training Episodes')
    ax.legend()

    fig.tight_layout()
    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    else:
        image_suffix = key.replace('/', '__')
        fig.savefig(f'./results_mpsc/{image_suffix}.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    ordered_models = ['none', 'none_cpen_0.01', 'none_cpen_0.1', 'none_cpen_1', 'mpsf_sr_pen_0.1', 'mpsf_sr_pen_1', 'mpsf_sr_pen_10', 'mpsf_sr_pen_100']

    colors = {
        'none': 'cornflowerblue',
        'none_cpen_0.01': 'plum',
        'none_cpen_0.1': 'mediumorchid',
        'none_cpen_1': 'darkorchid',
        'mpsf_sr_pen_0.1': 'lightgreen',
        'mpsf_sr_pen_1': 'limegreen',
        'mpsf_sr_pen_10': 'forestgreen',
        'mpsf_sr_pen_100': 'darkgreen',
    }

    def extract_rate_of_change_of_inputs(results_data, certified=True):
        return extract_rate_of_change(results_data, certified, order=1, mode='input')

    def extract_roc_cert(results_data, certified=True):
        return extract_rate_of_change_of_inputs(results_data, certified)

    def extract_roc_uncert(results_data, certified=False):
        return extract_rate_of_change_of_inputs(results_data, certified)

    def extract_rmse_cert(results_data, certified=True):
        return extract_rmse(results_data, certified)

    def extract_rmse_uncert(results_data, certified=False):
        return extract_rmse(results_data, certified)

    def extract_constraint_violations_cert(results_data, certified=True):
        return extract_constraint_violations(results_data, certified)

    def extract_constraint_violations_uncert(results_data, certified=False):
        return extract_constraint_violations(results_data, certified)

    def extract_reward_cert(results_data, certified=True):
        return extract_reward(results_data, certified)

    def extract_reward_uncert(results_data, certified=False):
        return extract_reward(results_data, certified)

    def extract_final_dist_cert(results_data, certified=True):
        return extract_final_dist(results_data, certified)

    def extract_final_dist_uncert(results_data, certified=False):
        return extract_final_dist(results_data, certified)

    def extract_failed_cert(results_data, certified=True):
        return extract_failed(results_data, certified)

    def extract_failed_uncert(results_data, certified=False):
        return extract_failed(results_data, certified)

    def extract_length_cert(results_data, certified=True):
        return extract_length(results_data, certified)

    def extract_length_uncert(results_data, certified=False):
        return extract_length(results_data, certified)

    system_name = 'quadrotor_3D'
    task_name = 'track'
    algo_name = 'ppo'
    if len(sys.argv) == 4:
        system_name = sys.argv[1]
        task_name = sys.argv[2]
        algo_name = sys.argv[3]

    plot_all_logs(system_name, task_name, algo_name)
    plot_step_time(system_name, task_name, algo_name)
    plot_model_comparisons(system_name, task_name, algo_name, extract_magnitude_of_corrections)
    plot_model_comparisons(system_name, task_name, algo_name, extract_percent_magnitude_of_corrections)
    plot_model_comparisons(system_name, task_name, algo_name, extract_max_correction)
    plot_model_comparisons(system_name, task_name, algo_name, extract_percent_max_correction)
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
    plot_model_comparisons(system_name, task_name, algo_name, extract_feasible_iterations)
    if task_name == 'stab':
        plot_model_comparisons(system_name, task_name, algo_name, extract_final_dist_cert)
        plot_model_comparisons(system_name, task_name, algo_name, extract_final_dist_uncert)
