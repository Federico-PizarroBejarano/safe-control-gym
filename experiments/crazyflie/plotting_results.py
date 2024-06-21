'''This script analyzes and plots the results from MPSC experiments. '''

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from safe_control_gym.safety_filters.mpsc.mpsc_utils import get_discrete_derivative
from safe_control_gym.utils.plotting import load_from_logs

Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)


plot = False
save_figs = True

suffix = '_dm'
# suffix = ''

ordered_models = [f'mpsf_0.1{suffix}', f'mpsf_1{suffix}', f'mpsf_10{suffix}', f'none{suffix}', f'none_cpen{suffix}']


def load_all_models(algo):
    '''Loads the results of every model for a specific experiment with every algo.

    Args:
        algo (str): The controller.

    Returns:
        all_results (dict): A dictionary containing all the results.
    '''

    all_results = {}

    if not CERTIFIED:
        is_cert = 'uncert'
    else:
        is_cert = 'cert'

    for model in ordered_models:
        if not REAL:
            with open(f'./results_cf/{algo}/{model}.pkl', 'rb') as f:
                all_results[model] = pickle.load(f)
        else:
            model_data = {'states': [],
                          'certified_action': [],
                          'corrections': [],
                          'uncertified_action': [],
                          'rewards': [],
                          'constraint_violations': [],
                          'length': [],
                          'rmse': [],
                          'feasible': []
                          }
            for test in range(len(os.listdir(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/{model}/{is_cert}'))):
                with open(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/{model}/{is_cert}/test{test}.pkl', 'rb') as file:
                    pickle_data = pickle.load(file)
                traj_goal = pickle_data['traj_goal']
                model_data['states'].append(pickle_data['states'])
                model_data['certified_action'].append(pickle_data['actions'])

                if not CERTIFIED:
                    model_data['corrections'].append(np.zeros(pickle_data['actions'].shape))
                else:
                    model_data['corrections'].append(pickle_data['corrections'])

                model_data['uncertified_action'].append(model_data['certified_action'][-1] - model_data['corrections'][-1])

                error = model_data['states'][-1][:, [0, 2]] - traj_goal[:, [0, 2]]
                dist = np.sum(2 * error * error, axis=1)
                reward = np.sum(np.exp(-dist))
                model_data['rewards'].append(reward)

                # TODO fix this
                constr_viols = np.sum(np.sum(np.abs(model_data['states'][-1][:, [0, 1, 2, 3, 6, 7]]) > np.array([[0.95, 2, 0.95, 2, 0.25, 0.25]]), axis=1) > 0)
                model_data['constraint_violations'].append(constr_viols)

                model_data['length'].append(len(model_data['states'][-1]))
                model_data['feasible'].append(len(model_data['states'][-1]))

                err = np.sum(model_data['states'][-1] * model_data['states'][-1], axis=1)
                rmse = np.mean(err)**0.5
                model_data['rmse'].append(rmse)

            all_results[model] = model_data

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


def extract_percent_magnitude_of_corrections(results_data):
    '''Extracts the percent magnitude of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        magn_of_corrections (list): The list of percent magnitude of corrections for all experiments.
    '''

    max_input = [np.maximum(np.linalg.norm(results_data['uncertified_action'][i], axis=1), np.linalg.norm(results_data['certified_action'][i], axis=1)) for i in range(len(results_data['uncertified_action']))]
    perc_change = [np.divide(np.linalg.norm(results_data['corrections'][i], axis=1), max_input[i]) for i in range(len(results_data['corrections']))]
    magn_of_corrections = [np.linalg.norm(elem) for elem in perc_change]

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


def extract_percent_max_correction(results_data):
    '''Extracts the percent max correction from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        max_corrections (list): The list of percent max corrections for all experiments.
    '''

    max_input = [np.maximum(np.linalg.norm(results_data['uncertified_action'][i], axis=1), np.linalg.norm(results_data['certified_action'][i], axis=1)) for i in range(len(results_data['uncertified_action']))]
    perc_change = [np.divide(np.linalg.norm(results_data['corrections'][i], axis=1), max_input[i]) for i in range(len(results_data['corrections']))]
    max_corrections = [np.max(elem) for elem in perc_change]

    return max_corrections


def extract_number_of_corrections(results_data):
    '''Extracts the number of corrections from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        num_corrections (list): The list of the number of corrections for all experiments.
    '''

    num_corrections = [np.sum(np.squeeze(np.abs(results_data['corrections'][i])) * 10 > np.squeeze(np.abs(results_data['certified_action'][i]))) for i in range(len(results_data['certified_action']))]
    return num_corrections


def extract_feasible_iterations(results_data):
    '''Extracts the number of feasible iterations from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        feasible_iterations (list): The list of the number of feasible iterations for all experiments.
    '''
    feasible_iterations = results_data['feasible']
    return feasible_iterations


def extract_rmse(results_data):
    '''Extracts the RMSEs from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        rmse (list): The list of RMSEs for all experiments.
    '''

    rmse = results_data['rmse']
    return rmse


def extract_length(results_data):
    '''Extracts the lengths from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        length (list): The list of lengths for all experiments.
    '''

    length = np.asarray([len(actions) for actions in results_data['uncertified_action']])
    return length


def extract_constraint_violations(results_data):
    '''Extracts the simulation time from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.
        certified (bool): Whether to extract the certified data or uncertified data.

    Returns:
        num_violations (list): The list of number of constraint violations for all experiments.
    '''

    num_violations = np.asarray(results_data['constraint_violations'])
    return num_violations


def extract_rate_of_change(results_data):
    '''Extracts the rate of change of a signal from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        roc (list): The list of rate of changes.
    '''
    all_signals = results_data['certified_action']

    total_derivatives = []
    for signal in all_signals:
        signal = signal.reshape(-1, 1)
        ctrl_freq = 25
        derivative = get_discrete_derivative(signal, ctrl_freq)
        total_derivatives.append(np.linalg.norm(derivative, 'fro'))

    return total_derivatives


def extract_reward(results_data):
    '''Extracts the mean reward from an experiment's data.

    Args:
        results_data (dict): A dictionary containing all the data from the desired experiment.

    Returns:
        mean_reward (list): The list of mean rewards.
    '''

    returns = np.asarray(results_data['rewards'])
    return returns


def create_paper_plot(data_extractor):
    '''Plots the constraint violations of every controller for a specific experiment.

    Args:
        data_extractor (lambda): A function that extracts the necessary data from the results.
    '''

    all_results = load_all_models('ppo')

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = ['Ours - 0.1', 'Ours - 1', 'Ours - 10', 'Std.', 'C.Pen.']
    colors = ['limegreen', 'forestgreen', 'darkgreen', 'cornflowerblue', 'pink']
    data = []

    for model in ordered_models:
        results = all_results[model]
        data.append(data_extractor(results))

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
        if REAL:
            if not CERTIFIED:
                is_cert = 'uncert'
            else:
                is_cert = 'cert'
            fig.savefig(f'./results_cf/{algo_name}/graphs/real_{is_cert}/{image_suffix}.png', dpi=300)
        else:
            if suffix != '':
                fig.savefig(f'./results_cf/{algo_name}/graphs/{suffix[1:]}/{image_suffix}.png', dpi=300)
            else:
                fig.savefig(f'./results_cf/{algo_name}/graphs/{image_suffix}.png', dpi=300)
            # tikzplotlib.save(f'./{image_suffix}.tex', axis_height='2.2in', axis_width='3.5in')
    if plot is True:
        plt.show()


def plot_all_logs(algo):
    '''Plots comparative plots of all the logs.

    Args:
        mpsc_cost_horizon (str): The cost horizon used by the smooth MPSC cost functions.
    '''
    all_results = {}

    for model in ordered_models:
        all_results[model] = []
        for seed in os.listdir(f'./models/rl_models/{algo}/{model}/'):
            if 'seed' in seed:
                all_results[model].append(load_from_logs(f'./models/rl_models/{algo}/{model}/{seed}/logs/'))

    for key in all_results[ordered_models[0]][0].keys():
        plot_log(algo, key, all_results)


def plot_log(algo, key, all_results):
    '''Plots a comparative plot of the log 'key'.

    Args:
        mpsc_cost_horizon (str): The cost horizon used by the smooth MPSC cost functions.
        key (str): The name of the log to be plotted.
        all_results (dict): A dictionary of all the logged results for all models.
    '''
    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)

    labels = [f'mpsf_0.1{suffix}', f'mpsf_1{suffix}', f'mpsf_10{suffix}', f'none{suffix}', f'none_cpen{suffix}']

    colors = {f'mpsf_0.1{suffix}': 'limegreen', f'mpsf_1{suffix}': 'forestgreen', f'mpsf_10{suffix}': 'darkgreen', f'none{suffix}': 'cornflowerblue', f'none_cpen{suffix}': 'plum'}

    names = {
        f'mpsf_0.1{suffix}': r'\textbf{Ours \$\boldsymbol{\alpha=10^{-1}}\$}',
        f'mpsf_1{suffix}': r'\textbf{Ours \$\boldsymbol{\alpha=10^0}\$}',
        f'mpsf_10{suffix}': r'\textbf{Ours \$\boldsymbol{\alpha=10^1}\$}',
        f'none{suffix}': 'Standard',
        f'none_cpen{suffix}': 'Constr. Pen.'
    }

    for model in labels:
        x = all_results[model][0][key][1] / 1000
        all_data = np.array([values[key][3] for values in all_results[model]])
        ax.plot(x, np.mean(all_data, axis=0), label=names[model], color=colors[model])
        ax.fill_between(x, np.min(all_data, axis=0), np.max(all_data, axis=0), alpha=0.3, edgecolor=colors[model], facecolor=colors[model])

    ylabels = {
        'stat_eval/ep_return': 'Episode Return',
        'stat/ep_constraint_violation': 'Episode Constraint Violations'
    }
    ax.set_ylabel(ylabels[key] if key in ylabels else key, weight='bold', fontsize=45, labelpad=10)
    ax.set_xlabel('Training Episodes')
    ax.legend()

    fig.tight_layout()
    ax.yaxis.grid(True)

    if plot is True:
        plt.show()
    if save_figs:
        image_suffix = key.replace('/', '__')
        if suffix != '':
            fig.savefig(f'./results_cf/{algo}/graphs/{suffix[1:]}/{image_suffix}.png', dpi=300)
        else:
            fig.savefig(f'./results_cf/{algo}/graphs/{image_suffix}.png', dpi=300)
            tikzplotlib.save(f'./{image_suffix}.tex', axis_height='2.2in', axis_width='3.5in')
    plt.close()


if __name__ == '__main__':
    REAL = True
    CERTIFIED = True
    algo_name = 'ppo'
    all_results = load_all_models(algo_name)

    create_paper_plot(extract_magnitude_of_corrections)
    create_paper_plot(extract_percent_magnitude_of_corrections)
    create_paper_plot(extract_max_correction)
    create_paper_plot(extract_percent_max_correction)
    create_paper_plot(extract_rate_of_change)
    create_paper_plot(extract_number_of_corrections)
    create_paper_plot(extract_feasible_iterations)
    create_paper_plot(extract_reward)
    create_paper_plot(extract_rmse)
    create_paper_plot(extract_constraint_violations)
    create_paper_plot(extract_length)

    if not REAL:
        plot_all_logs(algo_name)
