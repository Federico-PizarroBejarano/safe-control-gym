'''This script analyzes and plots the results from Subsystem experiments.'''

import pickle

import matplotlib.pyplot as plt
import numpy as np
from munch import munchify

from experiments.subsys.subsys_utils import (calculate_collisions, calculate_constraint_violations,
                                             calculate_input_rate_of_change, calculate_RMSE)

show_plots = True
save_plots = True

ordered_approaches = [
    'none_lqr',
    'none_mpc',
    'none',
    'naive',
    'safe_teleop_basic',
    'safe_teleop_advanced',
    'safe_swarm_basic',
    'safe_swarm_advanced',
    'ours'
]

all_labels = {
    'RMSE': 'RMSE (m)',
    'state_violations': 'State Violations',
    'input_violations': 'Input Violations',
    'collisions': 'Collisions',
    'rate_of_change_of_inputs': 'Rate of Change of Inputs',
    'mean_corrections': 'Mean Corrections',
    'max_corrections': 'Max Corrections',
    'time': 'Time (s)',
}

all_approach_labels = {
    'none_lqr': 'No SF (Only LQR)',
    'none_mpc': 'No SF (Only MPC)',
    'none': 'No SF',
    'naive': 'Naive SF',
    'safe_teleop_basic': 'Safe Teleop (Basic)',
    'safe_teleop_advanced': 'Safe Teleop (Advanced)',
    'safe_swarm_basic': 'Safe Swarm (Basic)',
    'safe_swarm_advanced': 'Safe Swarm (Advanced)',
    'ours': 'Our SF',
}


def load_all_approaches():
    '''Loads the results of every experiment.

    Returns:
        all_approaches (dict): A dictionary containing all the results.
    '''

    all_approaches = {}

    for approach in ordered_approaches:
        with open(f'./results/experiments/{approach}.pkl', 'rb') as f:
            all_approaches[approach] = munchify(pickle.load(f))

    return all_approaches


def extract_metric(data, key):
    '''Extracts the metric from the experiment data.

    Args:
        exp_data (dict): The experiment data.
        key (str): The key to be extracted.

    Returns:
        metric (float): The metric value.
    '''

    if key == 'RMSE':
        return np.mean(calculate_RMSE(data['obs'], data['X_goal'][:data['experiment_len'], :, :]))
    elif key == 'state_violations':
        return np.sum(calculate_constraint_violations(data['obs'], data['state_constraints']))
    elif key == 'input_violations':
        return np.sum(calculate_constraint_violations(data['actions'], data['input_constraints']))
    elif key == 'collisions':
        return np.sum(calculate_collisions(data['obs'], data['min_collision_distance']))
    elif key == 'rate_of_change_of_inputs':
        return np.mean(calculate_input_rate_of_change(data['actions'], data['frequency']))
    elif key == 'mean_corrections':
        return np.mean(data['corrections'])
    elif key == 'max_corrections':
        return np.max(data['corrections'])
    elif key == 'time':
        return data['time']
    else:
        raise ValueError(f'Invalid key: {key}')


def plot_all_results(all_results, key):
    '''Plots all the results.

    Args:
        all_results (dict): A dictionary containing all the results.
        key (str): The key to be plotted.
    '''

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    data = []

    for approach in ordered_approaches:
        exp_data = all_results[approach]
        data.append(extract_metric(exp_data, key))

    ylabel = all_labels[key]
    ax.set_ylabel(ylabel, weight='bold', fontsize=45, labelpad=10)

    x = np.arange(1, len(ordered_approaches) + 1)
    ax.set_xticks(x, [all_approach_labels[approach] for approach in ordered_approaches], weight='bold', fontsize=15, rotation=30, ha='right')

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(x)))  # Using viridis colormap from 0 to 0.8 for better visibility
    bars = ax.bar(x, data, width=0.75, color=colors)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

    fig.tight_layout()
    ax.set_ylim(ymin=0)
    ax.yaxis.grid(True)

    if save_plots:
        plt.savefig(f'./results/plots/{key}.png', dpi=300)
    if show_plots:
        plt.show()


if __name__ == '__main__':
    all_results = load_all_approaches()
    plot_all_results(all_results, 'RMSE')
    plot_all_results(all_results, 'state_violations')
    plot_all_results(all_results, 'input_violations')
    plot_all_results(all_results, 'collisions')
    plot_all_results(all_results, 'rate_of_change_of_inputs')
    plot_all_results(all_results, 'mean_corrections')
    plot_all_results(all_results, 'max_corrections')
    plot_all_results(all_results, 'time')
