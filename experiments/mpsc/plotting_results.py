'''This script analyzes and plots the results from MPSC experiments. '''

import pickle
from inspect import signature
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

from safe_control_gym.experiments.base_experiment import MetricExtractor
from safe_control_gym.envs.benchmark_env import Task, Environment
from safe_control_gym.safety_filters.mpsc.mpsc_utils import high_frequency_content, get_discrete_derivative


plot = True
save_figs = False
ordered_algos = ['lqr', 'pid', 'ppo', 'sac']

cost_colors = {'one_step':'cornflowerblue', 'constant': 'goldenrod', 'regularized': 'pink', 'lqr':'tomato', 'precomputed':'limegreen', 'learned':'yellow'}

U_EQs = {
    'cartpole': 0,
    'quadrotor_2D': 0.1323,
    'quadrotor_3D': 0.06615
}

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
    uncertified_data_std = []
    cert_data = defaultdict(list)
    cert_data_std = defaultdict(list)
    labels = []

    for algo in ordered_algos:
        if algo not in all_results:
            continue
        labels.append(algo.upper())

        for cost in ordered_costs:
            raw_data = all_results[algo][cost]
            cert_data[cost].append(data_extractor(raw_data)[0])
            cert_data_std[cost].append(data_extractor(raw_data)[1])
            if show_uncertified and cost == 'one_step':
                uncertified_data.append(data_extractor(raw_data, certified=False)[0])
                uncertified_data_std.append(data_extractor(raw_data, certified=False)[1])

    num_bars = len(ordered_costs)+show_uncertified
    width = 1/(num_bars+1)
    x = np.arange(len(labels))

    bars = {}

    if show_uncertified:
        bars['uncertified'] = ax.bar(x - (num_bars-1)/2.0*width, uncertified_data, yerr=uncertified_data_std, width=width, label='Uncertified', color='plum', capsize=10)

    for idx, cost in enumerate(ordered_costs):
        cost_name = cost.replace('_', ' ').title()
        if cost_name == 'Lqr':
            cost_name = 'LQR'
        position = ((num_bars-1)/2.0 - idx - show_uncertified)*width
        bars[cost] = ax.bar(x - position, cert_data[cost], yerr=cert_data_std[cost], width=width, label=f'{cost_name} Cost', color=cost_colors[cost], capsize=10)

    ylabel = data_extractor.__name__.replace('extract_', '').replace('_', ' ').title()
    if ylabel == 'Rmse':
        ylabel = 'RMSE'
    ax.set_ylabel(ylabel, weight='bold', fontsize=25, labelpad=10)
    task_title = 'Stabilization' if task == 'stab' else 'Trajectory Tracking'
    ax.set_title(f'{system.title()} {task_title} {ylabel} with M={mpsc_cost_horizon}', weight='bold', fontsize=25)

    ax.set_xticks(x, labels, weight='bold', fontsize=25)
    ax.legend(fontsize=25)

    if show_uncertified:
        rounded_labels = []
        for l in uncertified_data:
            if l > 100:
                rounded_labels.append(int(l))
            elif l > 1:
                rounded_labels.append(round(l, 1))
            else:
                rounded_labels.append(round(l, 2))

        ax.bar_label(bars['uncertified'], labels=rounded_labels, padding=3, fontsize=20)

    for cost in ordered_costs:
        rounded_labels = []
        for l in cert_data[cost]:
            if l > 100:
                rounded_labels.append(int(l))
            elif l > 1:
                rounded_labels.append(round(l, 1))
            else:
                rounded_labels.append(round(l, 2))
        ax.bar_label(bars[cost], labels=rounded_labels, padding=3, fontsize=20)

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
    data_std = []

    for algo in ordered_algos:
        if algo not in all_results:
            continue
        labels.append(algo.upper())

        one_step_cost = all_results[algo]['one_step']
        data.append(one_step_cost['uncert_metrics']['average_constraint_violation'])
        met = MetricExtractor(one_step_cost['uncert_results'])
        data_std.append(np.asarray(met.get_episode_constraint_violation_steps()).std())

    ax.set_ylabel('Number of Constraint Violations', weight='bold', fontsize=25, labelpad=10)
    task_title = 'Stabilization' if task == 'stab' else 'Trajectory Tracking'
    ax.set_title(f'{system.title()} {task_title} Constraint Violations with M={mpsc_cost_horizon}', weight='bold', fontsize=25)

    x = np.arange(len(labels))
    ax.set_xticks(x, labels, weight='bold', fontsize=25)

    cm = plt.cm.get_cmap('inferno', len(labels)+2)
    colors = [cm(i) for i in range(1, len(labels)+1)]
    violations = ax.bar(x, data, yerr=data_std, color=colors[::-1], capsize=10)
    ax.bar_label(violations, labels=data, padding=3, fontsize=20)

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    if save_figs:
        fig.savefig(f'./results_mpsc/{system}/{task}/m{mpsc_cost_horizon}/graphs/{system}_{task}_constraint_violations.png', dpi=300)


def extract_magnitude_of_correction(results_data):
    '''Extracts the mean correction from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        mean_magn_of_corrections (float): The mean magnitude of corrections for all experiments.
        std_magn_of_corrections (float): The standard deviation of the magnitude of corrections for all experiments.
    '''
    magn_of_corrections = [np.linalg.norm(mpsc_results['correction'][0]) for mpsc_results in results_data['cert_results']['safety_filter_data']]
    return np.mean(magn_of_corrections), np.std(magn_of_corrections)


def extract_max_correction(results_data):
    '''Extracts the max correction from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        mean_max_correction (float): The mean max correction for all experiments.
        std_max_corrections (float): The standard deviation of max corrections for all experiments.
    '''
    max_corrections = [np.max(np.abs(mpsc_results['correction'][0])) for mpsc_results in results_data['cert_results']['safety_filter_data']]

    return np.mean(max_corrections), np.std(max_corrections)


def extract_number_of_corrections(results_data):
    '''Extracts the number of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        mean_num_corrections (float): The mean number of corrections for all experiments.
        std_num_corrections (float): The standard deviation of the number of corrections for all experiments.
    '''
    num_corrections = [np.sum(mpsc_results['correction'][0]*10.0 > np.linalg.norm(results_data['cert_results']['current_clipped_action'][i] - U_EQs[system_name], axis=1)) for i, mpsc_results in enumerate(results_data['cert_results']['safety_filter_data'])]
    return np.mean(num_corrections), np.std(num_corrections)


def extract_rmse(results_data, certified=True):
    '''Extracts the number of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        mean_rmse (float): The mean RMSE for all experiments.
        std_rmse (float): The standard deviation of RMSE for all experiments.
    '''
    if certified:
        mean_rmse = results_data['cert_metrics']['average_rmse']
        std_rmse = results_data['cert_metrics']['rmse_std']
    else:
        mean_rmse = results_data['uncert_metrics']['average_rmse']
        std_rmse = results_data['uncert_metrics']['rmse_std']
    return mean_rmse, std_rmse


def extract_simulation_time(results_data, certified=True):
    '''Extracts the simulation time from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        mean_sim_time (float): The average simulation time for all experiments.
        std_sim_time (float): The standard deviation of the simulation time for all experiments.
    '''
    if certified:
        sim_time = [timestamp[-1] - timestamp[0] for timestamp in results_data['cert_results']['timestamp']]
    else:
        sim_time = [timestamp[-1] - timestamp[0] for timestamp in results_data['uncert_results']['timestamp']]

    return np.mean(sim_time), np.std(sim_time)


def extract_constraint_violations(results_data, certified=True):
    '''Extracts the simulation time from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        num_violations_mean (float): The average number of constraint violations for all experiments.
        num_violations_std (float): The standard deviation of the number of constraint violations for all experiments.
    '''
    if certified:
        num_violations_mean = results_data['cert_metrics']['average_constraint_violation']
        met = MetricExtractor(results_data['cert_results'])
        num_violations_std = np.asarray(met.get_episode_constraint_violation_steps()).std()
    else:
        num_violations_mean = results_data['uncert_metrics']['average_constraint_violation']
        met = MetricExtractor(results_data['uncert_results'])
        num_violations_std = np.asarray(met.get_episode_constraint_violation_steps()).std()

    return num_violations_mean, num_violations_std


def extract_high_frequency_content(results_data, certified=True):
    '''Extracts the high frequency content (HFC) from the inputs of an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        mean_HFC (float): The mean HFC for all experiments.
        std_HFC (float): The standard deviation of the HFC for all experiments.
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

    return np.mean(HFC), np.std(HFC)


def extract_rate_of_change(results_data, certified=True, order=1, mode='input'):
    '''Extracts the rate of change of a signal from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.
        order (int): Either 1 or 2, denoting the order of the derivative.
        mode (string): Either 'input' or 'correction', denoting which signal to use.

    Returns:
        mean_roc (float): The mean rate of change.
        std_roc (float): The standard deviation of the rates of change.
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
            if mode=='correction':
                signal = np.atleast_2d(signal).T
        elif n > 1:
            ctrl_freq = 50
        derivative = get_discrete_derivative(signal, ctrl_freq)
        if order == 2:
            derivative = get_discrete_derivative(derivative, ctrl_freq)
        total_derivatives.append(np.linalg.norm(derivative, 'fro'))

    return np.mean(total_derivatives), np.std(total_derivatives)


def extract_number_of_correction_intervals(results_data):
    '''Extracts the frequency the safety filter turns on or off from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        mean_num_correction_intervals (float): The mean number of times the filter starts correcting.
        std_num_correction_intervals (float): The standard deviation of the number of times the filter starts correcting.
    '''
    all_corrections = [(mpsc_results['correction'][0]*10.0 > np.linalg.norm(results_data['cert_results']['current_clipped_action'][i] - U_EQs[system_name], axis=1)) for i, mpsc_results in enumerate(results_data['cert_results']['safety_filter_data'])]

    correction_frequency = []
    for corrections in all_corrections:
        correction_frequency.append((np.diff(corrections)!=0).sum())

    return np.mean(correction_frequency), np.std(correction_frequency)


def plot_trajectories(config, X_GOAL, uncert_results, cert_results):
    '''Plots a series of graphs detailing the experiments in the passed in data.

    Args:
        config (dict): The configuration of the experiment.
        X_GOAL (np.ndarray): The goal (stabilization or reference trajectory) of the experiment.
        uncert_results (dict): The results of the uncertified experiment.
        cert_results (dict): The results of the certified experiment.
    '''
    met = MetricExtractor(cert_results)
    print('Total Certified Violations:', np.asarray(met.get_episode_constraint_violation_steps()).sum())

    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    for exp in range(len(uncert_results['obs'])):
        specific_results = {key:[cert_results[key][exp]] for key in cert_results.keys()}
        met = MetricExtractor(specific_results)
        print(f'Total Certified Violations ({exp}):', np.asarray(met.get_episode_constraint_violation_steps()).sum())
        mpsc_results = cert_results['safety_filter_data'][exp]
        corrections = mpsc_results['correction'][0]*10.0 > np.linalg.norm(cert_results['current_clipped_action'][exp] - U_EQs[system], axis=1)
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
        ax.plot(cert_results['obs'][exp][:,graph1_1], cert_results['obs'][exp][:,graph1_2],'.-', label='Certified')
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
            ax2.plot(np.linspace(0, 20, cert_results['obs'][exp].shape[0])[1:], X_GOAL[:,0],'g--', label='Reference')
            ax2.plot(np.linspace(0, 20, uncert_results['obs'][exp].shape[0]), uncert_results['obs'][exp][:,0],'r--', label='Uncertified')
            ax2.plot(np.linspace(0, 20, cert_results['obs'][exp].shape[0]), cert_results['obs'][exp][:,0],'.-', label='Certified')
            ax2.plot(np.linspace(0, 20, cert_results['obs'][exp].shape[0])[corrections], cert_results['obs'][exp][corrections, 0], 'r.', label='Modified')
            ax2.set_xlabel(r'Time')
            ax2.set_ylabel(r'X')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')
        elif config.task == Environment.QUADROTOR:
            _, ax2 = plt.subplots()
            ax2.plot(uncert_results['obs'][exp][:,graph3_1+1], uncert_results['obs'][exp][:,graph3_2+1],'r--', label='Uncertified')
            ax2.plot(cert_results['obs'][exp][:,graph3_1+1], cert_results['obs'][exp][:,graph3_2+1],'.-', label='Certified')
            ax2.plot(cert_results['obs'][exp][corrections, graph3_1+1], cert_results['obs'][exp][corrections, graph3_2+1], 'r.', label='Modified')
            ax2.set_xlabel(r'x_dot')
            ax2.set_ylabel(r'z_dot')
            ax2.set_box_aspect(0.5)
            ax2.legend(loc='upper right')

        _, ax3 = plt.subplots()
        ax3.plot(uncert_results['obs'][exp][:,graph3_1], uncert_results['obs'][exp][:,graph3_2],'r--', label='Uncertified')
        ax3.plot(cert_results['obs'][exp][:,graph3_1], cert_results['obs'][exp][:,graph3_2],'.-', label='Certified')
        if config.task_config.task == Task.TRAJ_TRACKING and config.task == Environment.QUADROTOR:
            ax3.plot(X_GOAL[:,graph3_1], X_GOAL[:,graph3_2],'g--', label='Reference')
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

        _, ax_fft = plt.subplots()
        N_cert = max(cert_results['current_clipped_action'][exp].shape)
        N_uncert = max(uncert_results['current_clipped_action'][exp].shape)
        if config.task == Environment.CARTPOLE:
            spectrum_cert = fft(np.squeeze(cert_results['current_clipped_action'][exp]))
            spectrum_uncert = fft(np.squeeze(uncert_results['current_clipped_action'][exp]))
            freq_cert = fftfreq(len(spectrum_cert), 1/config.task_config['ctrl_freq'])[:N_cert//2]
            freq_uncert = fftfreq(len(spectrum_uncert), 1/config.task_config['ctrl_freq'])[:N_uncert//2]
            ax_fft.plot(freq_cert, 2.0/N_cert * np.abs(spectrum_cert[0:N_cert//2]), 'b-', label='Certified')
            ax_fft.plot(freq_uncert, 2.0/N_uncert * np.abs(spectrum_uncert[0:N_uncert//2]), 'r--', label='Uncertified')
        else:
            spectrum_cert1 = fft(np.squeeze(cert_results['current_clipped_action'][exp][:, 0]))
            spectrum_cert2 = fft(np.squeeze(cert_results['current_clipped_action'][exp][:, 1]))
            spectrum_uncert1 = fft(np.squeeze(uncert_results['current_clipped_action'][exp][:, 0]))
            spectrum_uncert2 = fft(np.squeeze(uncert_results['current_clipped_action'][exp][:, 1]))
            freq_cert1 = fftfreq(len(spectrum_cert1), 1/config.task_config['ctrl_freq'])[:N_cert//2]
            freq_cert2 = fftfreq(len(spectrum_cert2), 1/config.task_config['ctrl_freq'])[:N_cert//2]
            freq_uncert1 = fftfreq(len(spectrum_uncert1), 1/config.task_config['ctrl_freq'])[:N_uncert//2]
            freq_uncert2 = fftfreq(len(spectrum_uncert2), 1/config.task_config['ctrl_freq'])[:N_uncert//2]
            ax_fft.plot(freq_cert1, 2.0/N_cert * np.abs(spectrum_cert1[0:N_cert//2]), 'b-', label='Certified 1')
            ax_fft.plot(freq_cert2, 2.0/N_cert * np.abs(spectrum_cert2[0:N_cert//2]), 'b-', label='Certified 2')
            ax_fft.plot(freq_uncert1, 2.0/N_uncert * np.abs(spectrum_uncert1[0:N_uncert//2]), 'r--', label='Uncertified 1')
            ax_fft.plot(freq_uncert2, 2.0/N_uncert * np.abs(spectrum_uncert2[0:N_uncert//2]), 'r--', label='Uncertified 2')

        ax_fft.legend()
        ax_fft.set_title('Fourier Analysis of Inputs')
        ax_fft.set_xlabel('Frequency')
        ax_fft.set_ylabel('Magnitude')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    mpsc_cost_horizon_num = 2

    if mpsc_cost_horizon_num == 2:
        ordered_costs = ['one_step', 'constant', 'regularized', 'lqr', 'precomputed', 'learned']
    else:
        ordered_costs = ['one_step', 'regularized', 'lqr', 'precomputed', 'learned']

    def extract_1st_order_rate_of_change_of_inputs(results_data, certified=True): return extract_rate_of_change(results_data, certified, order=1, mode='input')
    def extract_2nd_order_rate_of_change_of_inputs(results_data, certified=True): return extract_rate_of_change(results_data, certified, order=2, mode='input')
    def extract_1st_order_rate_of_change_of_corrections(results_data): return extract_rate_of_change(results_data, certified=True, order=1, mode='correction')
    def extract_2nd_order_rate_of_change_of_corrections(results_data): return extract_rate_of_change(results_data, certified=True, order=2, mode='correction')

    for system_name in ['cartpole', 'quadrotor_2D', 'quadrotor_3D']:
        for task_name in ['stab', 'track']:
            plot_violations(system_name, task_name, mpsc_cost_horizon_num)

            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_1st_order_rate_of_change_of_inputs)
            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_2nd_order_rate_of_change_of_inputs)
            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_1st_order_rate_of_change_of_corrections)
            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_2nd_order_rate_of_change_of_corrections)

            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_high_frequency_content)

            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_magnitude_of_correction)
            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_max_correction)

            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_simulation_time)
            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_rmse)

            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_number_of_corrections)
            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_number_of_correction_intervals)

            plot_experiment(system_name, task_name, mpsc_cost_horizon_num, extract_constraint_violations)

    # # Plotting a single experiment
    # algo_name = 'lqr'
    # mpsc_cost_name = 'one_step'
    # one_result = load_one_experiment(system=system_name, task=task_name, algo=algo_name, mpsc_cost_horizon=mpsc_cost_horizon_num)
    # results = one_result[mpsc_cost_name]
    # plot_trajectories(results['config'], results['X_GOAL'], results['uncert_results'], results['cert_results'])
