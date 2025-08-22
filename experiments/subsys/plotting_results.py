'''This script analyzes and plots the results from Subsystem experiments.'''

import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from munch import munchify

from experiments.subsys.subsys_utils import (calculate_collisions, calculate_constraint_violations,
                                             calculate_input_rate_of_change, calculate_RMSE)

show_plots = False
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
    'RMSE_teleop': 'RMSE (Teleop) (m)',
    'RMSE_swarm': 'RMSE (Swarm) (m)',
    'state_violations': 'State Violations',
    'input_violations': 'Input Violations',
    'collisions': 'Collisions',
    'rate_of_change_of_inputs': 'Rate of Change of Inputs',
    'mean_corrections': 'Mean Corrections',
    'max_corrections': 'Max Corrections',
    'time': 'Time to Execute (s)',
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

all_state_labels = [
    'X (m)',
    'X vel (m/s)',
    'Y (m)',
    'Y vel (m/s)',
    'Z (m)',
    'Z vel (m/s)',
    'phi (rad)',
    'theta (rad)',
    'psi (rad)',
    'p (rad/s)',
    'q (rad/s)',
    'r (rad/s)',
]


def plot_trajectory_2D(approach_name, all_obs, X_goal, indices, state_constraints):
    # Get constraint bounds from safety filter config
    upper_bounds = np.array(state_constraints.upper_bounds)[[indices]].squeeze()
    lower_bounds = np.array(state_constraints.lower_bounds)[[indices]].squeeze()

    # Plot trajectory and constraints
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    colors = ['r', 'g', 'b', 'y']
    for drone_idx in range(all_obs.shape[1]):
        ax.plot(all_obs[:, drone_idx, indices[0]], all_obs[:, drone_idx, indices[1]], label=f'Trajectory_{drone_idx}', color=colors[drone_idx])
        ax.plot(X_goal[:, drone_idx, indices[0]], X_goal[:, drone_idx, indices[1]], label=f'Goal_{drone_idx}', color=colors[drone_idx], linestyle=(0, (5, 5)))

    ax.fill_between([lower_bounds[0], upper_bounds[0]], [lower_bounds[1], lower_bounds[1]], [upper_bounds[1], upper_bounds[1]], color='r', alpha=0.1)

    ax.set_title(all_approach_labels[approach_name], weight='bold', fontsize=20)
    ax.set_xlabel(all_state_labels[indices[0]], weight='bold', fontsize=15)
    ax.set_ylabel(all_state_labels[indices[1]], weight='bold', fontsize=15)
    plt.grid(True)
    plt.legend()
    if save_plots:
        plt.savefig(f'./results/plots/trajectories/{approach_name}.png', dpi=300)
    if show_plots:
        plt.show()


def plot_trajectory_3D(all_obs, X_goal, state_constraints):
    # Extract position data
    positions = all_obs[:, :, [0, 2, 4]]
    x = positions[:, :, 0]
    y = positions[:, :, 1]
    z = positions[:, :, 2]

    # Get constraint bounds from safety filter config
    x_bounds = [state_constraints.lower_bounds[0], state_constraints.upper_bounds[0]]
    y_bounds = [state_constraints.lower_bounds[2], state_constraints.upper_bounds[2]]
    z_bounds = [state_constraints.lower_bounds[4], state_constraints.upper_bounds[4]]

    # Plot trajectory and constraints
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y']
    for drone_idx in range(positions.shape[1]):
        ax.plot(x[:, drone_idx], y[:, drone_idx], z[:, drone_idx], label=f'Trajectory_{drone_idx}', color=colors[drone_idx])
        ax.plot(X_goal[:, drone_idx, 0], X_goal[:, drone_idx, 2], X_goal[:, drone_idx, 4], label=f'Goal_{drone_idx}', color=colors[drone_idx], linestyle=(0, (5, 5)))

    add_box_to_plot(ax, x_bounds, y_bounds, z_bounds)

    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

    # Plot velocity data
    # Extract velocity data
    velocities = all_obs[:, :, [1, 3, 5]]
    x = velocities[:, :, 0]
    y = velocities[:, :, 1]
    z = velocities[:, :, 2]

    # Get constraint bounds from safety filter config
    x_bounds = [state_constraints.lower_bounds[1], state_constraints.upper_bounds[1]]
    y_bounds = [state_constraints.lower_bounds[3], state_constraints.upper_bounds[3]]
    z_bounds = [state_constraints.lower_bounds[5], state_constraints.upper_bounds[5]]

    # Plot trajectory and constraints
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y']
    for drone_idx in range(velocities.shape[1]):
        ax.plot(x[:, drone_idx], y[:, drone_idx], z[:, drone_idx], label=f'Velocity_{drone_idx}', color=colors[drone_idx])

    add_box_to_plot(ax, x_bounds, y_bounds, z_bounds)

    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()


def add_box_to_plot(ax, x_bounds, y_bounds, z_bounds):
    # Create data for the box faces
    xx, yy = np.meshgrid([x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]])
    z1 = np.ones_like(xx) * z_bounds[0]
    z2 = np.ones_like(xx) * z_bounds[1]

    # Plot the 6 faces of the box
    ax.plot_surface(xx, yy, z1, alpha=0.1, color='r')  # Bottom
    ax.plot_surface(xx, yy, z2, alpha=0.1, color='r')  # Top

    yy, zz = np.meshgrid([y_bounds[0], y_bounds[1]], [z_bounds[0], z_bounds[1]])
    x1 = np.ones_like(yy) * x_bounds[0]
    x2 = np.ones_like(yy) * x_bounds[1]
    ax.plot_surface(x1, yy, zz, alpha=0.1, color='r')  # Left
    ax.plot_surface(x2, yy, zz, alpha=0.1, color='r')  # Right

    xx, zz = np.meshgrid([x_bounds[0], x_bounds[1]], [z_bounds[0], z_bounds[1]])
    y1 = np.ones_like(xx) * y_bounds[0]
    y2 = np.ones_like(xx) * y_bounds[1]
    ax.plot_surface(xx, y1, zz, alpha=0.1, color='r')  # Front
    ax.plot_surface(xx, y2, zz, alpha=0.1, color='r')  # Back


def create_video(frames, fps, name):
    size = 480, 640
    out = cv2.VideoWriter(f'./results/videos/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    for frame in frames:
        out.write(frame)
    out.release()


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

    if key == 'RMSE_teleop':
        rmse = calculate_RMSE(data['obs'], data['X_goal'][:data['experiment_len'], :, :])[data['teleop_vec']]
        return np.mean(rmse)
    elif key == 'RMSE_swarm':
        rmse = calculate_RMSE(data['obs'], data['X_goal'][:data['experiment_len'], :, :])[~data['teleop_vec']]
        return np.mean(rmse)
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
    ax.set_xlim(xmin=0.2)
    ax.yaxis.grid(True)

    if save_plots:
        plt.savefig(f'./results/plots/{key}.png', dpi=300)
    if show_plots:
        plt.show()


if __name__ == '__main__':
    all_results = load_all_approaches()

    # Plot metrics
    plot_all_results(all_results, 'RMSE_teleop')
    plot_all_results(all_results, 'RMSE_swarm')
    plot_all_results(all_results, 'state_violations')
    plot_all_results(all_results, 'input_violations')
    plot_all_results(all_results, 'collisions')
    plot_all_results(all_results, 'rate_of_change_of_inputs')
    plot_all_results(all_results, 'mean_corrections')
    plot_all_results(all_results, 'max_corrections')
    plot_all_results(all_results, 'time')

    # Plot trajectories
    for approach in ordered_approaches:
        exp_data = all_results[approach]
        plot_trajectory_2D(approach, exp_data['obs'], exp_data['X_goal'], [0, 2], exp_data['state_constraints'])
