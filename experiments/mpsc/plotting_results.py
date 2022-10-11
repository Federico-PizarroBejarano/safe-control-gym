'''This script analyzes and plots the results from MPSC experiments. '''

import pickle
from inspect import signature

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq


ordered_algos = ['lqr', 'pid', 'ppo', 'sac']


def load_one_experiment(system, task, algo):
    '''Loads the results of every MPSC cost function for a specific experiment.

    Args:
        system (str): The system to be controlled.
        task (str): The task to be completed (either 'stab' or 'track').
        algo (str): The controller being used.

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    with open(f'./results/{system}/{task}/results_{system}_{task}_{algo}_one_step_cost.pkl', 'rb') as f:
        one_step_cost_results = pickle.load(f)

    with open(f'./results/{system}/{task}/results_{system}_{task}_{algo}_lqr_cost.pkl', 'rb') as f:
        lqr_cost_results = pickle.load(f)

    with open(f'./results/{system}/{task}/results_{system}_{task}_{algo}_precomputed_cost.pkl', 'rb') as f:
        precomputed_cost_results = pickle.load(f)

    with open(f'./results/{system}/{task}/results_{system}_{task}_{algo}_learned_cost.pkl', 'rb') as f:
        learned_cost_results = pickle.load(f)

    all_results = {'one_step_cost': one_step_cost_results,
                   'lqr_cost': lqr_cost_results,
                   'precomputed_cost': precomputed_cost_results,
                   'learned_cost': learned_cost_results}

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
    one_step_data = []
    lqr_data = []
    precomputed_data = []
    learned_data = []
    labels = []

    for algo in ordered_algos:
        if algo not in all_results.keys():
            continue
        labels.append(algo.upper())

        one_step_cost = all_results[algo]['one_step_cost']
        lqr_cost = all_results[algo]['lqr_cost']
        precomputed_cost = all_results[algo]['precomputed_cost']
        learned_cost = all_results[algo]['learned_cost']

        if show_uncertified:
            uncertified_data.append(data_extractor(one_step_cost, certified=False))
        one_step_data.append(data_extractor(one_step_cost))
        lqr_data.append(data_extractor(lqr_cost))
        precomputed_data.append(data_extractor(precomputed_cost))
        learned_data.append(data_extractor(learned_cost))

    width = 1/(5+show_uncertified)
    x = np.arange(len(labels))

    if show_uncertified:
        uncertified = ax.bar(x - 2*width, uncertified_data, width, label='Uncertified', color='plum')
        one_step = ax.bar(x - width, one_step_data, width, label='One Step Cost', color='cornflowerblue')
        lqr = ax.bar(x, lqr_data, width, label='LQR Cost', color='tomato')
        precomputed = ax.bar(x + width, precomputed_data, width, label='Precomputed Cost', color='limegreen')
        learned = ax.bar(x + 2*width, learned_data, width, label='Learned Cost', color='yellow')
    else:
        one_step = ax.bar(x - 3*width/2, one_step_data, width, label='One Step Cost', color='cornflowerblue')
        lqr = ax.bar(x - width/2, lqr_data, width, label='LQR Cost', color='tomato')
        precomputed = ax.bar(x + width/2, precomputed_data, width, label='Precomputed Cost', color='limegreen')
        learned = ax.bar(x + 3*width/2, learned_data, width, label='Learned Cost', color='yellow')

    ylabel = data_extractor.__name__.replace('extract_', '').replace('_', ' ').title()
    ax.set_ylabel(ylabel, weight='bold', fontsize=25, labelpad=10)

    ax.set_xticks(x, labels, weight='bold', fontsize=25)
    ax.legend(fontsize=25)

    if show_uncertified:
        ax.bar_label(uncertified, labels=np.round(uncertified_data, 1), padding=3, fontsize=20)
    ax.bar_label(one_step, labels=np.round(one_step_data, 1), padding=3, fontsize=20)
    ax.bar_label(lqr, labels=np.round(lqr_data, 1), padding=3, fontsize=20)
    ax.bar_label(precomputed, labels=np.round(precomputed_data, 1), padding=3, fontsize=20)
    ax.bar_label(learned, labels=np.round(learned_data, 1), padding=3, fontsize=20)

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)
    # plt.show()

    image_suffix = data_extractor.__name__.replace('extract_', '')
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
        if algo not in all_results.keys():
            continue
        labels.append(algo.upper())

        one_step_cost = all_results[algo]['one_step_cost']
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
    # plt.show()

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
    system = 'cartpole'
    task = 'track'
    plot_violations(system=system, task=task)
    plot_experiment(system=system, task=task, data_extractor=extract_magnitude_of_correction)
    plot_experiment(system=system, task=task, data_extractor=extract_max_correction)
    plot_experiment(system=system, task=task, data_extractor=extract_high_frequency_content)
    plot_experiment(system=system, task=task, data_extractor=extract_simulation_time)
    plot_experiment(system=system, task=task, data_extractor=extract_rmse)
    plot_experiment(system=system, task=task, data_extractor=extract_number_of_corrections)
