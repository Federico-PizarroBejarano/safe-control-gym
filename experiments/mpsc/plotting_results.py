'''This script analyzes and plots the results from MPSC experiments. '''

import pickle
from inspect import signature
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq


plot = True
save_figs = False
ordered_algos = ['lqr', 'pid', 'ppo', 'sac']
ordered_costs = ['one_step', 'lqr', 'precomputed', 'learned']

cost_colors = {'one_step':'cornflowerblue', 'lqr':'tomato', 'precomputed':'limegreen', 'learned':'yellow'}


def load_one_experiment(system, task, algo):
    '''Loads the results of every MPSC cost function for a specific experiment.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
        algo (str): The controller being used.

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    all_results = {}

    for cost in ordered_costs:
        with open(f'./results/{system}/{task}/results_{system}_{task}_{algo}_{cost}_cost.pkl', 'rb') as f:
            all_results[cost] = pickle.load(f)

    return all_results


def load_all_algos(system, task):
    '''Loads the results of every MPSC cost function for a specific experiment with every algo.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    all_results = {}

    for algo in ['lqr', 'pid', 'ppo', 'sac']:
        if system == 'cartpole' and algo == 'pid':
            continue

        all_results[algo] = load_one_experiment(system, task, algo)

    return all_results


def plot_experiment(system, task, data_extractor):
    '''Plots the results of every MPSC cost function for a specific experiment.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
        data_extractor (lambda): A function that extracts the necessary data from the results.
    '''

    all_results = load_all_algos(system, task)

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

    num_bars = len(ordered_costs)+show_uncertified
    width = 1/(num_bars+1)
    x = np.arange(len(labels))

    bars = {}

    if show_uncertified:
        bars['uncertified'] = ax.bar(x - (num_bars-1)/2.0*width, uncertified_data, width, label='Uncertified', color='plum')

    for idx, cost in enumerate(ordered_costs):
        cost_name = cost.replace('_', ' ').title()
        if cost_name == 'Lqr':
            cost_name = 'LQR'
        position = ((num_bars-1)/2.0 - idx - show_uncertified)*width
        bars[cost] = ax.bar(x - position, cert_data[cost], width, label=f'{cost_name} Cost', color=cost_colors[cost])

    ylabel = data_extractor.__name__.replace('extract_', '').replace('_', ' ').title()
    if ylabel == 'Rmse':
        ylabel = 'RMSE'
    ax.set_ylabel(ylabel, weight='bold', fontsize=25, labelpad=10)

    ax.set_xticks(x, labels, weight='bold', fontsize=25)
    ax.legend(fontsize=25)

    if show_uncertified:
        ax.bar_label(bars['uncertified'], labels=np.round(uncertified_data, 1), padding=3, fontsize=20)

    for cost in ordered_costs:
        ax.bar_label(bars[cost], labels=np.round(cert_data[cost], 1), padding=3, fontsize=20)

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)
    if plot is True:
        plt.show()

    image_suffix = data_extractor.__name__.replace('extract_', '')
    if save_figs:
        fig.savefig(f'./results/{system}/{task}/graphs/{system}_{task}_{image_suffix}.png', dpi=300)


def plot_violations(system, task):
    '''Plots the constraint violations of every controller for a specific experiment.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
    '''

    all_results = load_all_algos(system, task)

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = []
    data = []

    for algo in ordered_algos:
        if algo not in all_results:
            continue
        labels.append(algo.upper())

        one_step_cost = all_results[algo]['one_step']
        data.append(one_step_cost['uncert_metrics']['average_constraint_violation'])

    ax.set_ylabel('Number of Constraint Violations', weight='bold', fontsize=25, labelpad=10)

    x = np.arange(len(labels))
    ax.set_xticks(x, labels, weight='bold', fontsize=25)

    cm = plt.cm.get_cmap('inferno', len(labels)+2)
    colors = [cm(i) for i in range(1, len(labels)+1)]
    violations = ax.bar(x, data, color=colors[::-1])
    ax.bar_label(violations, labels=data, padding=3, fontsize=20)

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    if save_figs:
        fig.savefig(f'./results/{system}/{task}/graphs/{system}_{task}_constraint_violations.png', dpi=300)


def extract_magnitude_of_correction(results_data):
    '''Extracts the mean correction from an experiment's data.

    Args:
        magnitude_of_correction (float): The mean magnitude of corrections for all experiments.
    '''
    return np.mean([np.linalg.norm(mpsc_results['correction'][0]) for mpsc_results in results_data['cert_results']['safety_filter_data']])


def extract_max_correction(results_data):
    '''Extracts the max correction from an experiment's data.

    Args:
        max_correction (float): The mean max correction for all experiments.
    '''
    return np.mean([np.max(np.abs(mpsc_results['correction'][0])) for mpsc_results in results_data['cert_results']['safety_filter_data']])


def extract_number_of_corrections(results_data):
    '''Extracts the number of corrections from an experiment's data.

    Args:
        num_corrections (float): The mean number of correction for all experiments.
    '''
    return np.mean([np.sum(mpsc_results['correction'][0] > 1e-4) for mpsc_results in results_data['cert_results']['safety_filter_data']])


def extract_rmse(results_data, certified=True):
    '''Extracts the number of corrections from an experiment's data.

    Args:
        num_corrections (float): The mean number of correction for all experiments.
    '''
    if certified:
        rmse = results_data['cert_metrics']['average_rmse']
    else:
        rmse = results_data['uncert_metrics']['average_rmse']
    return rmse


def extract_simulation_time(results_data, certified=True):
    '''Extracts the simulation time from an experiment's data.

    Args:
        simulation_time (float): The mean number of iterations for all experiments.
    '''
    if certified:
        simulation_time = np.mean([timestamp[-1] - timestamp[0] for timestamp in results_data['cert_results']['timestamp']])
    else:
        simulation_time = np.mean([timestamp[-1] - timestamp[0] for timestamp in results_data['uncert_results']['timestamp']])
    return simulation_time


def extract_high_frequency_content(results_data, certified=True):
    '''Extracts the high frequency content (HFC) from the inputs of an experiment's data.

    Args:
        HFC (float): The mean HFC for all experiments.
    '''
    N = max(results_data['cert_results']['current_physical_action'][0].shape)
    n = min(results_data['cert_results']['current_physical_action'][0].shape)

    if certified:
        all_actions = results_data['cert_results']['current_physical_action']
    else:
        all_actions = results_data['uncert_results']['current_physical_action']

    HFC = 0
    for actions in all_actions:
        if n == 1:
            ctrl_freq = 15
            spectrum = fft(np.squeeze(actions))
            freq = fftfreq(len(spectrum), 1/ctrl_freq)[:N//2]
            HFC += freq.T @ (2.0/N * np.abs(spectrum[0:N//2]))
        elif n > 1:
            ctrl_freq = 50
            for i in range(n):
                spectrum = fft(np.squeeze(actions[:, i]))
                freq = fftfreq(len(spectrum), 1/ctrl_freq)[:N//2]
                HFC += freq.T @ (2.0/N * np.abs(spectrum[0:N//2]))

    return HFC/len(all_actions)


if __name__ == '__main__':
    system_name = 'cartpole'
    task_name = 'track'
    plot_violations(system=system_name, task=task_name)
    plot_experiment(system=system_name, task=task_name, data_extractor=extract_magnitude_of_correction)
    plot_experiment(system=system_name, task=task_name, data_extractor=extract_max_correction)
    plot_experiment(system=system_name, task=task_name, data_extractor=extract_high_frequency_content)
    plot_experiment(system=system_name, task=task_name, data_extractor=extract_simulation_time)
    plot_experiment(system=system_name, task=task_name, data_extractor=extract_rmse)
    plot_experiment(system=system_name, task=task_name, data_extractor=extract_number_of_corrections)
