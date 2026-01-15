'''Run ablation experiments for adversarial reward function.

This script:
1. Loads trained models from ablation/<system>/<experiment>/seed_<N>/
2. Runs evaluation with safety filter
3. Logs phase plots (theta vs theta_dot) for analysis

Supports both cartpole and quadrotor_2D environments.
'''

import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

# Cartpole experiments
CARTPOLE_EXPERIMENTS = [
    # 'full_reward',
    # 'correction_only',
    'state_only',
    'no_correction_mag',
    # 'no_correction_ratio',
    'no_correction_bonus',
    # 'no_correction_penalty',
    # 'no_theta',
    'no_velocity',
    # 'no_oscillation',
    'no_stability_penalty',
    # 'no_cart_penalty',
    'w_correction_10',
    # 'w_correction_50',
    # 'temp_5',
    'temp_30',
    'temp_500',
    # 'no_safe_reset',
]

# Quadrotor 2D experiments
QUADROTOR_2D_EXPERIMENTS = [
    'state_only',
    'no_correction_mag',
    'no_correction_bonus',
    'no_velocity',
    'no_stability_penalty',
    'no_altitude_penalty',
    'no_position_reward',
    'w_correction_10',
    'temp_30',
    'temp_500',
]

# System configurations
SYSTEM_CONFIGS = {
    'cartpole': {
        'task': 'cartpole',
        'algo': 'ppo',
        'safety_filter': 'nl_mpsc',
        'config_files': [
            'config_overrides/cartpole/ppo_cartpole.yaml',
            'config_overrides/cartpole/cartpole_track.yaml',
            'config_overrides/cartpole/nl_mpsc_cartpole.yaml',
        ],
        'experiments': CARTPOLE_EXPERIMENTS,
        'output_subdir': '',  # cartpole ablations are in ablation/ directly
        # State indices for plotting
        'theta_idx': 2,
        'theta_dot_idx': 3,
        'pos_idx': 0,
        'vel_idx': 1,
        'constraint_theta_idx': 2,  # Index in constraint bounds
        # Labels
        'theta_label': r'$\theta$ (rad)',
        'theta_dot_label': r'$\dot{\theta}$ (rad/s)',
        'pos_label': r'$x$ (m)',
        'vel_label': r'$\dot{x}$ (m/s)',
        'pos2_idx': None,  # No second position for cartpole
    },
    'quadrotor_2D': {
        'task': 'quadrotor',
        'algo': 'ppo',
        'safety_filter': 'nl_mpsc',
        'config_files': [
            'config_overrides/quadrotor_2D/ppo_quadrotor_2D.yaml',
            'config_overrides/quadrotor_2D/quadrotor_2D_track.yaml',
            'config_overrides/quadrotor_2D/nl_mpsc_quadrotor_2D.yaml',
        ],
        'experiments': QUADROTOR_2D_EXPERIMENTS,
        'output_subdir': 'quadrotor_2D',  # quadrotor ablations are in ablation/quadrotor_2D/
        # State indices: [x, x_dot, z, z_dot, theta, theta_dot]
        'theta_idx': 4,
        'theta_dot_idx': 5,
        'pos_idx': 0,  # x position
        'vel_idx': 1,  # x velocity
        'pos2_idx': 2,  # z position (for trajectory plot)
        'constraint_theta_idx': 4,  # Index in constraint bounds
        # Labels
        'theta_label': r'$\theta$ (rad)',
        'theta_dot_label': r'$\dot{\theta}$ (rad/s)',
        'pos_label': r'$x$ (m)',
        'vel_label': r'$z$ (m)',
    },
}

SEEDS = [2]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'ablation')


def evaluate_and_plot(exp_name, model_path, seed, save_dir, system='cartpole'):
    '''Evaluate a trained model and save phase plots.

    Args:
        exp_name: Name of the experiment
        model_path: Path to the trained model
        seed: Random seed used for training
        save_dir: Directory to save outputs
        system: System type ('cartpole' or 'quadrotor_2D')
    '''
    print(f'Evaluating: {exp_name} (seed={seed}, system={system})')

    sys_config = SYSTEM_CONFIGS[system]

    # Build config using ConfigFactory
    config_files = [os.path.join(SCRIPT_DIR, f) for f in sys_config['config_files']]

    sys.argv = [
        '',
        '--task', sys_config['task'],
        '--algo', sys_config['algo'],
        '--safety_filter', sys_config['safety_filter'],
        '--overrides', *config_files,
    ]

    fac = ConfigFactory()
    config = fac.merge()

    # Create environment
    env_func = partial(make, config.task, **config.task_config)
    env = env_func()

    # Setup controller and load model
    ctrl = make(config.algo, env_func, **config.algo_config, output_dir=save_dir)
    ctrl.load(model_path)

    # Setup safety filter
    config.task_config['normalized_rl_action_space'] = False
    env_func_filter = partial(make, config.task, **config.task_config)
    safety_filter = make(config.safety_filter, env_func_filter, **config.sf_config)
    safety_filter.reset()
    ctrl.reset()

    if config.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
        safety_filter.cost_function.uncertified_controller = ctrl

    # Run evaluation with safety filter
    experiment = BaseExperiment(env, ctrl, safety_filter=safety_filter)
    cert_results, cert_metrics = experiment.run_evaluation(n_episodes=1)

    # Run evaluation without safety filter for comparison
    ctrl.reset()
    experiment_uncert = BaseExperiment(env, ctrl)
    uncert_results, uncert_metrics = experiment_uncert.run_evaluation(n_episodes=1)

    ctrl.close()
    safety_filter.close()

    # Extract data
    mpsc_results = cert_results['safety_filter_data']
    corrections = mpsc_results['correction'][0] > 1e-6
    # Handle multi-dimensional corrections (quadrotor has 2 inputs)
    if corrections.ndim > 1:
        corrections = np.any(corrections, axis=1)
    corrections = np.append(corrections, False)

    # Get constraint value
    theta_constraint = config.task_config['constraints'][0].upper_bounds[sys_config['constraint_theta_idx']]

    # Get state indices
    theta_idx = sys_config['theta_idx']
    theta_dot_idx = sys_config['theta_dot_idx']
    pos_idx = sys_config['pos_idx']
    pos2_idx = sys_config['pos2_idx']

    # Create phase plot (theta vs theta_dot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Phase portrait (theta vs theta_dot)
    ax1 = axes[0]
    ax1.plot(uncert_results['obs'][0][:, theta_idx], uncert_results['obs'][0][:, theta_dot_idx],
             'r--', alpha=0.5, label='Uncertified')
    ax1.plot(cert_results['obs'][0][:, theta_idx], cert_results['obs'][0][:, theta_dot_idx],
             'b-', linewidth=1.5, label='Certified')
    ax1.plot(cert_results['obs'][0][corrections, theta_idx], cert_results['obs'][0][corrections, theta_dot_idx],
             'r.', markersize=4, label='Corrections')
    ax1.scatter(cert_results['obs'][0][0, theta_idx], cert_results['obs'][0][0, theta_dot_idx],
                color='g', marker='o', s=100, zorder=5, label='Start')
    ax1.axvline(x=-theta_constraint, color='k', lw=2, linestyle='--', label='Constraint')
    ax1.axvline(x=theta_constraint, color='k', lw=2, linestyle='--')
    ax1.set_xlabel(sys_config['theta_label'], fontsize=12)
    ax1.set_ylabel(sys_config['theta_dot_label'], fontsize=12)
    ax1.set_title(f'Phase Portrait: {exp_name}', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Position plot (different for cartpole vs quadrotor)
    ax2 = axes[1]
    t = np.arange(len(cert_results['obs'][0][:, theta_idx]))

    if system == 'cartpole':
        # Time series of theta and theta_dot
        ax2.plot(t, cert_results['obs'][0][:, theta_idx], 'b-', label=r'$\theta$')
        ax2.plot(t, cert_results['obs'][0][:, theta_dot_idx], 'g-', label=r'$\dot{\theta}$')
        ax2.axhline(y=-theta_constraint, color='k', lw=1, linestyle='--', alpha=0.5)
        ax2.axhline(y=theta_constraint, color='k', lw=1, linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time step', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title(f'Time Series: {exp_name}', fontsize=14)
    else:
        # For quadrotor: x-z trajectory plot
        ax2.plot(uncert_results['obs'][0][:, pos_idx], uncert_results['obs'][0][:, pos2_idx],
                 'r--', alpha=0.5, label='Uncertified')
        ax2.plot(cert_results['obs'][0][:, pos_idx], cert_results['obs'][0][:, pos2_idx],
                 'b-', linewidth=1.5, label='Certified')
        ax2.plot(cert_results['obs'][0][corrections, pos_idx], cert_results['obs'][0][corrections, pos2_idx],
                 'r.', markersize=4, label='Corrections')
        ax2.scatter(cert_results['obs'][0][0, pos_idx], cert_results['obs'][0][0, pos2_idx],
                    color='g', marker='o', s=100, zorder=5, label='Start')
        # Plot reference trajectory if available
        if hasattr(safety_filter.env, 'X_GOAL'):
            ax2.plot(safety_filter.env.X_GOAL[:, 0], safety_filter.env.X_GOAL[:, 2],
                     'g--', alpha=0.7, label='Reference')
        ax2.set_xlabel(sys_config['pos_label'], fontsize=12)
        ax2.set_ylabel(sys_config['vel_label'], fontsize=12)
        ax2.set_title(f'Trajectory (x-z): {exp_name}', fontsize=14)

    # Mark corrections on time axis (for cartpole time series)
    if system == 'cartpole':
        correction_times = np.where(corrections[:-1])[0]
        for ct in correction_times:
            ax2.axvline(x=ct, color='r', alpha=0.1, lw=1)

    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, 'phase_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save trajectory data for later plotting
    trajectory_data = {
        'theta': cert_results['obs'][0][:, theta_idx],
        'theta_dot': cert_results['obs'][0][:, theta_dot_idx],
        'theta_uncert': uncert_results['obs'][0][:, theta_idx],
        'theta_dot_uncert': uncert_results['obs'][0][:, theta_dot_idx],
        'corrections': corrections[:-1],
        'theta_constraint': theta_constraint,
        'system': system,
    }

    # Add position data for quadrotor
    if system == 'quadrotor_2D':
        trajectory_data['x'] = cert_results['obs'][0][:, pos_idx]
        trajectory_data['z'] = cert_results['obs'][0][:, pos2_idx]
        trajectory_data['x_uncert'] = uncert_results['obs'][0][:, pos_idx]
        trajectory_data['z_uncert'] = uncert_results['obs'][0][:, pos2_idx]

    np.savez(os.path.join(save_dir, 'trajectory_data.npz'), **trajectory_data)

    # Compute metrics
    metrics = {
        'experiment': exp_name,
        'seed': seed,
        'system': system,
        'num_corrections': int(np.sum(corrections)),
        'total_correction_magnitude': float(np.linalg.norm(mpsc_results['correction'][0])),
        'max_correction': float(np.max(np.abs(mpsc_results['correction'][0]))),
        'avg_correction': float(np.mean(np.abs(mpsc_results['correction'][0]))),
        'episode_length_cert': int(cert_metrics['average_length']),
        'episode_length_uncert': int(uncert_metrics['average_length']),
        'constraint_violations_cert': int(cert_metrics['average_constraint_violation']),
        'constraint_violations_uncert': int(uncert_metrics['average_constraint_violation']),
        'rmse_cert': float(cert_metrics['average_rmse']),
        'rmse_uncert': float(uncert_metrics['average_rmse']),
        'theta_max': float(np.max(np.abs(cert_results['obs'][0][:, theta_idx]))),
        'theta_dot_max': float(np.max(np.abs(cert_results['obs'][0][:, theta_dot_idx]))),
    }

    print(f'  Corrections: {metrics["num_corrections"]}, '
          f'Total: {metrics["total_correction_magnitude"]:.2f}, '
          f'Theta max: {metrics["theta_max"]:.3f}')

    return metrics


def create_summary_plot(all_metrics, output_dir, system='cartpole'):
    '''Create summary comparison plots across all experiments.

    Args:
        all_metrics: List of metric dictionaries
        output_dir: Directory to save the summary plot
        system: System type for labeling
    '''
    # Filter metrics for this system
    system_metrics = [m for m in all_metrics if m.get('system', 'cartpole') == system]

    if not system_metrics:
        print(f'No metrics found for system: {system}')
        return

    # Get unique experiment names
    exp_names = []
    for m in system_metrics:
        if m['experiment'] not in exp_names:
            exp_names.append(m['experiment'])

    # Aggregate metrics across seeds
    summary = {}
    for exp_name in exp_names:
        exp_metrics = [m for m in system_metrics if m['experiment'] == exp_name]
        if exp_metrics:
            summary[exp_name] = {
                'num_corrections_mean': np.mean([m['num_corrections'] for m in exp_metrics]),
                'num_corrections_std': np.std([m['num_corrections'] for m in exp_metrics]),
                'total_correction_mean': np.mean([m['total_correction_magnitude'] for m in exp_metrics]),
                'total_correction_std': np.std([m['total_correction_magnitude'] for m in exp_metrics]),
                'theta_max_mean': np.mean([m['theta_max'] for m in exp_metrics]),
                'theta_max_std': np.std([m['theta_max'] for m in exp_metrics]),
            }

    # Create bar plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    exp_labels = list(summary.keys())
    x = np.arange(len(exp_labels))
    width = 0.6

    # Plot 1: Number of corrections
    ax1 = axes[0]
    means = [summary[e]['num_corrections_mean'] for e in exp_labels]
    stds = [summary[e]['num_corrections_std'] for e in exp_labels]
    ax1.bar(x, means, width, yerr=stds, capsize=3, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Number of Corrections', fontsize=12)
    ax1.set_title(f'Safety Filter Corrections per Episode ({system})', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Total correction magnitude
    ax2 = axes[1]
    means = [summary[e]['total_correction_mean'] for e in exp_labels]
    stds = [summary[e]['total_correction_std'] for e in exp_labels]
    ax2.bar(x, means, width, yerr=stds, capsize=3, color='darkorange', alpha=0.8)
    ax2.set_ylabel('Total Correction Magnitude', fontsize=12)
    ax2.set_title(f'Cumulative Correction Magnitude ({system})', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Max theta achieved
    ax3 = axes[2]
    means = [summary[e]['theta_max_mean'] for e in exp_labels]
    stds = [summary[e]['theta_max_std'] for e in exp_labels]
    ax3.bar(x, means, width, yerr=stds, capsize=3, color='forestgreen', alpha=0.8)
    ax3.set_ylabel('Max |θ| (rad)', fontsize=12)
    ax3.set_title(f'Maximum Angle Achieved ({system})', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Determine save path based on system
    if system == 'cartpole':
        summary_path = os.path.join(output_dir, 'ablation_summary.png')
    else:
        summary_path = os.path.join(output_dir, SYSTEM_CONFIGS[system]['output_subdir'], 'ablation_summary.png')

    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSummary plot saved to: {summary_path}')


def plot_all_phase_plots(output_dir, experiments=None, seeds=None, system='cartpole'):
    '''Plot phase plots for all experiments from saved trajectory data.

    Args:
        output_dir: Base output directory
        experiments: List of experiment names (default: use system config)
        seeds: List of seeds to include
        system: System type ('cartpole' or 'quadrotor_2D')
    '''
    sys_config = SYSTEM_CONFIGS[system]

    if experiments is None:
        experiments = sys_config['experiments']
    if seeds is None:
        seeds = SEEDS

    # Determine the data directory
    if sys_config['output_subdir']:
        data_dir = os.path.join(output_dir, sys_config['output_subdir'])
    else:
        data_dir = output_dir

    # Collect all trajectory data
    trajectories = []
    for exp_name in experiments:
        for seed in seeds:
            npz_path = os.path.join(data_dir, exp_name, f'seed_{seed}', 'trajectory_data.npz')
            if os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                traj_data = {
                    'experiment': exp_name,
                    'seed': seed,
                    'theta': data['theta'],
                    'theta_dot': data['theta_dot'],
                    'theta_uncert': data['theta_uncert'],
                    'theta_dot_uncert': data['theta_dot_uncert'],
                    'corrections': data['corrections'],
                    'theta_constraint': float(data['theta_constraint']),
                }
                # Load quadrotor-specific data if available
                if 'x' in data:
                    traj_data['x'] = data['x']
                    traj_data['z'] = data['z']
                    traj_data['x_uncert'] = data['x_uncert']
                    traj_data['z_uncert'] = data['z_uncert']
                trajectories.append(traj_data)

    if not trajectories:
        print(f'No trajectory data found for {system}. Run evaluation first.')
        return

    # Get unique experiments
    exp_names = []
    for t in trajectories:
        if t['experiment'] not in exp_names:
            exp_names.append(t['experiment'])

    # Create grid of phase plots
    n_exp = len(exp_names)
    n_cols = min(4, n_exp)
    n_rows = (n_exp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_exp == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, exp_name in enumerate(exp_names):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Get all seeds for this experiment
        exp_trajs = [t for t in trajectories if t['experiment'] == exp_name]
        theta_constraint = exp_trajs[0]['theta_constraint']

        # Plot each seed with different alpha
        for i, traj in enumerate(exp_trajs):
            alpha = 0.5 if len(exp_trajs) > 1 else 1.0
            ax.plot(traj['theta'], traj['theta_dot'], '-', alpha=alpha,
                    linewidth=1, label=f'seed {traj["seed"]}' if i == 0 else None)

            # Mark corrections
            corr_idx = np.where(traj['corrections'])[0]
            if len(corr_idx) > 0:
                ax.plot(traj['theta'][corr_idx], traj['theta_dot'][corr_idx],
                        'r.', markersize=2, alpha=0.5)

        # Add constraint lines
        ax.axvline(x=-theta_constraint, color='k', lw=1.5, linestyle='--')
        ax.axvline(x=theta_constraint, color='k', lw=1.5, linestyle='--')

        ax.set_xlabel(r'$\theta$', fontsize=10)
        ax.set_ylabel(r'$\dot{\theta}$', fontsize=10)
        ax.set_title(exp_name, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.3, 0.3)

    # Hide empty subplots
    for idx in range(n_exp, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f'Phase Plots: {system}', fontsize=14, y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(data_dir, 'all_phase_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'All phase plots saved to: {plot_path}')

    # For quadrotor, also create trajectory plots
    if system == 'quadrotor_2D' and any('x' in t for t in trajectories):
        plot_all_trajectory_plots(trajectories, exp_names, data_dir)

    return plot_path


def plot_all_trajectory_plots(trajectories, exp_names, output_dir):
    '''Plot x-z trajectory plots for quadrotor experiments.

    Args:
        trajectories: List of trajectory data dictionaries
        exp_names: List of experiment names
        output_dir: Directory to save the plot
    '''
    n_exp = len(exp_names)
    n_cols = min(4, n_exp)
    n_rows = (n_exp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_exp == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, exp_name in enumerate(exp_names):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Get all seeds for this experiment
        exp_trajs = [t for t in trajectories if t['experiment'] == exp_name]

        # Plot each seed
        for i, traj in enumerate(exp_trajs):
            if 'x' not in traj:
                continue
            alpha = 0.5 if len(exp_trajs) > 1 else 1.0
            ax.plot(traj['x'], traj['z'], 'b-', alpha=alpha, linewidth=1,
                    label=f'Certified (seed {traj["seed"]})' if i == 0 else None)
            ax.plot(traj['x_uncert'], traj['z_uncert'], 'r--', alpha=alpha * 0.5,
                    linewidth=1, label='Uncertified' if i == 0 else None)

            # Mark corrections
            corr_idx = np.where(traj['corrections'])[0]
            if len(corr_idx) > 0:
                ax.plot(traj['x'][corr_idx], traj['z'][corr_idx],
                        'r.', markersize=2, alpha=0.5)

        ax.set_xlabel(r'$x$ (m)', fontsize=10)
        ax.set_ylabel(r'$z$ (m)', fontsize=10)
        ax.set_title(exp_name, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    # Hide empty subplots
    for idx in range(n_exp, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('Trajectory Plots (x-z): quadrotor_2D', fontsize=14, y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'all_trajectory_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'All trajectory plots saved to: {plot_path}')


def save_metrics_csv(all_metrics, output_dir, system='cartpole'):
    '''Save all metrics to a CSV file.

    Args:
        all_metrics: List of metric dictionaries
        output_dir: Base output directory
        system: System type for file naming
    '''
    import csv

    # Filter metrics for this system
    system_metrics = [m for m in all_metrics if m.get('system', 'cartpole') == system]

    if not system_metrics:
        print(f'No metrics to save for system: {system}')
        return

    # Determine save path based on system
    if system == 'cartpole':
        csv_path = os.path.join(output_dir, 'ablation_metrics.csv')
    else:
        csv_path = os.path.join(output_dir, SYSTEM_CONFIGS[system]['output_subdir'], 'ablation_metrics.csv')

    keys = system_metrics[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(system_metrics)

    print(f'Metrics saved to: {csv_path}')


def main():
    '''Main function to run all ablation experiments.'''
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate ablation experiments')
    parser.add_argument('--system', type=str, default='cartpole',
                        choices=['cartpole', 'quadrotor_2D'],
                        help='System to evaluate (default: cartpole)')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='Specific experiments to run (default: all for the system)')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Seeds to use (default: 2)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot from existing trajectory data (no evaluation)')
    args = parser.parse_args()

    system = args.system
    sys_config = SYSTEM_CONFIGS[system]

    # Filter experiments if specified, otherwise use system defaults
    experiments = args.experiments if args.experiments else sys_config['experiments']

    # Determine output directory for this system
    if sys_config['output_subdir']:
        system_output_dir = os.path.join(OUTPUT_DIR, sys_config['output_subdir'])
    else:
        system_output_dir = OUTPUT_DIR

    print('\n' + '#' * 60)
    print('# ADVERSARIAL REWARD ABLATION EVALUATION')
    print(f'# System: {system}')
    print(f'# Experiments: {len(experiments)}')
    print(f'# Seeds: {args.seeds}')
    print(f'# Output: {system_output_dir}')
    print('#' * 60 + '\n')

    # Plot-only mode: just create phase plots from saved data
    if args.plot_only:
        plot_all_phase_plots(OUTPUT_DIR, experiments, args.seeds, system=system)
        print('\n' + '#' * 60)
        print('# PLOTTING COMPLETE')
        print('#' * 60 + '\n')
        return

    all_metrics = []

    for exp_name in experiments:
        for seed in args.seeds:
            model_dir = os.path.join(system_output_dir, exp_name, f'seed_{seed}')
            model_path = os.path.join(model_dir, 'model_latest.pt')

            if not os.path.exists(model_path):
                print(f'[SKIP] {exp_name} (seed={seed}) - model not found at {model_path}')
                continue

            try:
                metrics = evaluate_and_plot(exp_name, model_path, seed, model_dir, system=system)
                all_metrics.append(metrics)
            except Exception as e:
                print(f'[ERROR] {exp_name} (seed={seed}): {e}')
                import traceback
                traceback.print_exc()

    # Create summary plots and save metrics
    if all_metrics:
        create_summary_plot(all_metrics, OUTPUT_DIR, system=system)
        save_metrics_csv(all_metrics, OUTPUT_DIR, system=system)
        plot_all_phase_plots(OUTPUT_DIR, experiments, args.seeds, system=system)

    print('\n' + '#' * 60)
    print('# EVALUATION COMPLETE')
    print(f'# Results saved to: {system_output_dir}')
    print('#' * 60 + '\n')


if __name__ == '__main__':
    main()
