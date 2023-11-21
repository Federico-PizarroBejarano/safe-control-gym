'''This script analyzes and plots the results from MPSC experiments. '''

from safe_control_gym.safety_filters.mpsc.mpsc_utils import get_discrete_derivative
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)


k

plot = True
save_figs = False

ordered_costs = ['one_step', 'regularized', 'precomputed M=2', 'precomputed M=5', 'precomputed M=10']


def load_all_algos():
    '''Loads all the results.

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    all_results = {}
    num_tests = 5

    for cost in ordered_costs:
        all_results[cost] = {'states': [], 'inputs': [], 'corrections': []}
        if 'precomputed' in cost:
            cost_name = 'precomputed'
            M = cost.split('=')[1]
            extra_folder = f'm{M}/'
        else:
            cost_name = cost
            extra_folder = ''

        for test in range(num_tests):
            all_results[cost]['states'].append(np.load(f'./all_trajs/test{test}/cert/{cost_name}_cost/{extra_folder}states.npy'))
            all_results[cost]['inputs'].append(np.load(f'./all_trajs/test{test}/cert/{cost_name}_cost/{extra_folder}actions.npy'))
            all_results[cost]['corrections'].append(np.load(f'./all_trajs/test{test}/cert/{cost_name}_cost/{extra_folder}corrections.npy'))

    all_results['uncert'] = {'states': [], 'inputs': []}
    for test in range(num_tests):
        all_results['uncert']['states'].append(np.load(f'./all_trajs/test{test}/uncert/states.npy'))
        all_results['uncert']['inputs'].append(np.load(f'./all_trajs/test{test}/uncert/actions.npy'))

    return all_results


def extract_magnitude_of_corrections(results_data):
    '''Extracts the magnitude of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        magn_of_corrections (list): The list of magnitude of corrections for all experiments.
    '''

    magn_of_corrections = [np.linalg.norm(results_data['corrections'][i]) for i in range(len(results_data['corrections']))]
    return magn_of_corrections


def extract_max_correction(results_data):
    '''Extracts the max correction from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        max_corrections (list): The list of max corrections for all experiments.
    '''
    max_corrections = [np.max(np.abs(results_data['corrections'][i])) for i in range(len(results_data['corrections']))]

    return max_corrections


def extract_rate_of_change(results_data, mode='input'):
    '''Extracts the rate of change of a signal from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        mode (string): Either 'input' or 'correction', denoting which signal to use.

    Returns:
        roc (list): The list of rate of changes.
    '''
    if mode == 'input':
        all_signals = results_data['inputs']
    elif mode == 'correction':
        all_signals = results_data['corrections']

    total_derivatives = []
    for signal in all_signals:
        signal = signal.reshape(-1, 1)
        ctrl_freq = 25
        derivative = get_discrete_derivative(signal, ctrl_freq)
        total_derivatives.append(np.linalg.norm(derivative, 'fro'))

    return total_derivatives


def extract_number_of_corrections(results_data):
    '''Extracts the number of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        num_corrections (list): The list of the number of corrections for all experiments.
    '''

    num_corrections = [np.sum(np.squeeze(np.abs(results_data['corrections'][i])) * 10 > np.squeeze(np.abs(results_data['inputs'][i]))) for i in range(len(results_data['inputs']))]
    return num_corrections


def create_paper_plot(data_extractor):
    '''Plots the constraint violations of every controller for a specific experiment.

    Args:
        data_extractor (lambda): A function that extracts the necessary data from the results.
    '''

    all_results = load_all_algos()

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = [r'$J_{MPSF, 1}$', r'$+J_{reg, 10}$', r'$J_{MPSF, 2}$', r'$J_{MPSF, 5}$', r'$J_{MPSF, 10}$']
    colors = ['cornflowerblue', 'pink', 'limegreen', 'forestgreen', 'darkgreen']
    data = []

    for cost in ordered_costs:
        if cost == 'one_step' and data_extractor == extract_rate_of_change_of_inputs:
            labels = ['Uncert'] + labels
            colors = ['plum'] + colors
            results = all_results['uncert']
            data.append(data_extractor(results))
        results = all_results[cost]
        data.append(data_extractor(results))
        print(f'Cost: {cost}, val:{np.mean(data_extractor(results))}')

    ylabel = data_extractor.__name__.replace('extract_', '').replace('_', ' ').title()
    ax.set_ylabel(ylabel, weight='bold', fontsize=35, labelpad=10)

    x = np.arange(1, len(labels) + 1)
    ax.set_xticks([])

    medianprops = dict(linestyle='--', linewidth=2.5, color='black')
    flierprops = {'marker': 'o', 'markersize': 3}
    bplot = ax.boxplot(data, patch_artist=True, medianprops=medianprops, flierprops=flierprops, widths=[0.75] * len(labels))
    ax.set_xticks(x, labels, fontsize=25)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    fig.tight_layout()

    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)

    image_suffix = data_extractor.__name__.replace('extract_', '')
    if save_figs:
        fig.savefig(f'./all_trajs/{image_suffix}.png', dpi=300)
        tikzplotlib.save(f'./all_trajs/{image_suffix}.tex', axis_height='2.2in', axis_width='3.5in')
    if plot is True:
        plt.show()


def create_chattering_plot(TEST=0):
    all_results = load_all_algos()

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = ['Uncert', r'$J_{MPSF, 1}$', r'$J_{MPSF, 10}$']
    colors = ['plum', 'cornflowerblue', 'darkgreen']

    start = 7
    freq = 25

    uncert_states = all_results['uncert']['states'][TEST][start * freq:, 0]
    one_step_states = all_results['one_step']['states'][TEST][start * freq:, 0]
    precomputed_states = all_results['precomputed M=10']['states'][TEST][start * freq:, 0]

    one_step_corr = all_results['one_step']['corrections'][TEST][start * freq:]
    one_step_inputs = all_results['one_step']['inputs'][TEST][start * freq:]
    precomputed_corr = all_results['precomputed M=10']['corrections'][TEST][start * freq:]
    precomputed_inputs = all_results['precomputed M=10']['inputs'][TEST][start * freq:]

    ax.plot(np.array(range(len(uncert_states))) / freq + start, uncert_states, label=labels[0], color=colors[0], linewidth=2)
    ax.plot(np.array(range(len(one_step_states))) / freq + start, one_step_states, label=labels[1], color=colors[1], linewidth=2)
    ax.plot(np.array(range(len(precomputed_states))) / freq + start, precomputed_states, label=labels[2], color=colors[2], linewidth=2)

    corrections_one_step = np.squeeze(np.abs(one_step_corr)) * 10 > np.squeeze(np.abs(one_step_inputs))
    corrections_precomputed = np.squeeze(np.abs(precomputed_corr)) * 10 > np.squeeze(np.abs(precomputed_inputs))

    corr_one_step_ranges = get_ranges(corrections_one_step)
    corr_precomputed_ranges = get_ranges(corrections_precomputed)

    for r in corr_one_step_ranges:
        ax.plot((np.array(range(len(uncert_states))) / freq)[r[0]:r[1]] + start, one_step_states[r[0]:r[1]], color=colors[1], linewidth=8, alpha=0.5)

    for r in corr_precomputed_ranges:
        ax.plot((np.array(range(len(precomputed_states))) / freq)[r[0]:r[1]] + start, precomputed_states[r[0]:r[1]], color=colors[2], linewidth=8, alpha=0.5)

    fig.tight_layout()
    ax.yaxis.grid(True)

    ax.set_ylabel('x Position [m]', weight='bold', fontsize=35, labelpad=10)
    ax.set_xlabel('Time [s]', weight='bold', fontsize=35, labelpad=10)

    ax.legend(loc='upper left', fontsize=25)

    if save_figs:
        fig.savefig('./all_trajs/traj.png', dpi=300)
        tikzplotlib.save('./all_trajs/traj.tex', axis_height='2.2in', axis_width='3.3in')
    if plot is True:
        plt.show()


def get_ranges(l):
    ranges = []

    started_range = False
    for i, val in enumerate(l):
        if val == True and started_range is False:
            min_idx = i
            started_range = True
        if val == False and started_range is True:
            ranges.append((min_idx, i - 1))
            started_range = False

    if val == True and started_range is True:
        ranges.append((min_idx, i))

    return ranges


if __name__ == '__main__':
    create_chattering_plot(TEST=3)
    def extract_rate_of_change_of_inputs(results_data): return extract_rate_of_change(results_data, mode='input')

    # create_paper_plot(extract_magnitude_of_corrections)
    # create_paper_plot(extract_max_correction)
    # create_paper_plot(extract_rate_of_change_of_inputs)
    # create_paper_plot(extract_number_of_corrections)
