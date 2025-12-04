'''Run ablation experiments for adversarial reward function.

This script:
1. Loads trained models from ablation/<experiment>/seed_<N>/
2. Runs evaluation with safety filter
3. Logs phase plots (theta vs theta_dot) for analysis
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
EXPERIMENTS = [
    'full_reward',
    'correction_only',
    'state_only',
    'no_correction_mag',
    'no_correction_ratio',
    'no_correction_bonus',
    'no_correction_penalty',
    'no_theta',
    'no_velocity',
    'no_oscillation',
    'no_stability_penalty',
    'no_cart_penalty',
    'w_correction_10',
    'w_correction_50',
    'temp_5',
    'temp_30',
    'no_safe_reset',
]

SEEDS = [42, 62, 821]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'ablation')


def evaluate_and_plot(exp_name, model_path, seed, save_dir):
    '''Evaluate a trained model and save phase plots.'''
    print(f'Evaluating: {exp_name} (seed={seed})')

    # Build config using ConfigFactory
    config_files = [
        os.path.join(SCRIPT_DIR, 'config_overrides/cartpole/ppo_cartpole.yaml'),
        os.path.join(SCRIPT_DIR, 'config_overrides/cartpole/cartpole_track.yaml'),
        os.path.join(SCRIPT_DIR, 'config_overrides/cartpole/nl_mpsc_cartpole.yaml'),
    ]

    sys.argv = [
        '',
        '--task', 'cartpole',
        '--algo', 'ppo',
        '--safety_filter', 'nl_mpsc',
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
    corrections = np.append(corrections, False)

    theta_constraint = config.task_config['constraints'][0].upper_bounds[2]

    # Create phase plot (theta vs theta_dot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Phase portrait (theta vs theta_dot)
    ax1 = axes[0]
    ax1.plot(uncert_results['obs'][0][:, 2], uncert_results['obs'][0][:, 3],
             'r--', alpha=0.5, label='Uncertified')
    ax1.plot(cert_results['obs'][0][:, 2], cert_results['obs'][0][:, 3],
             'b-', linewidth=1.5, label='Certified')
    ax1.plot(cert_results['obs'][0][corrections, 2], cert_results['obs'][0][corrections, 3],
             'r.', markersize=4, label='Corrections')
    ax1.scatter(cert_results['obs'][0][0, 2], cert_results['obs'][0][0, 3],
                color='g', marker='o', s=100, zorder=5, label='Start')
    ax1.axvline(x=-theta_constraint, color='k', lw=2, linestyle='--', label='Constraint')
    ax1.axvline(x=theta_constraint, color='k', lw=2, linestyle='--')
    ax1.set_xlabel(r'$\theta$ (rad)', fontsize=12)
    ax1.set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
    ax1.set_title(f'Phase Portrait: {exp_name}', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time series of theta and theta_dot
    ax2 = axes[1]
    t = np.arange(len(cert_results['obs'][0][:, 2]))
    ax2.plot(t, cert_results['obs'][0][:, 2], 'b-', label=r'$\theta$')
    ax2.plot(t, cert_results['obs'][0][:, 3], 'g-', label=r'$\dot{\theta}$')
    ax2.axhline(y=-theta_constraint, color='k', lw=1, linestyle='--', alpha=0.5)
    ax2.axhline(y=theta_constraint, color='k', lw=1, linestyle='--', alpha=0.5)

    # Mark corrections on time axis
    correction_times = np.where(corrections[:-1])[0]
    for ct in correction_times:
        ax2.axvline(x=ct, color='r', alpha=0.1, lw=1)

    ax2.set_xlabel('Time step', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title(f'Time Series: {exp_name}', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, 'phase_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Save trajectory data for later plotting
    trajectory_data = {
        'theta': cert_results['obs'][0][:, 2],
        'theta_dot': cert_results['obs'][0][:, 3],
        'theta_uncert': uncert_results['obs'][0][:, 2],
        'theta_dot_uncert': uncert_results['obs'][0][:, 3],
        'corrections': corrections[:-1],
        'theta_constraint': theta_constraint,
    }
    np.savez(os.path.join(save_dir, 'trajectory_data.npz'), **trajectory_data)

    # Compute metrics
    metrics = {
        'experiment': exp_name,
        'seed': seed,
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
        'theta_max': float(np.max(np.abs(cert_results['obs'][0][:, 2]))),
        'theta_dot_max': float(np.max(np.abs(cert_results['obs'][0][:, 3]))),
    }

    print(f'  Corrections: {metrics["num_corrections"]}, '
          f'Total: {metrics["total_correction_magnitude"]:.2f}, '
          f'Theta max: {metrics["theta_max"]:.3f}')

    return metrics


def create_summary_plot(all_metrics, output_dir):
    '''Create summary comparison plots across all experiments.'''
    # Get unique experiment names
    exp_names = []
    for m in all_metrics:
        if m['experiment'] not in exp_names:
            exp_names.append(m['experiment'])

    # Aggregate metrics across seeds
    summary = {}
    for exp_name in exp_names:
        exp_metrics = [m for m in all_metrics if m['experiment'] == exp_name]
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
    ax1.set_title('Safety Filter Corrections per Episode', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Total correction magnitude
    ax2 = axes[1]
    means = [summary[e]['total_correction_mean'] for e in exp_labels]
    stds = [summary[e]['total_correction_std'] for e in exp_labels]
    ax2.bar(x, means, width, yerr=stds, capsize=3, color='darkorange', alpha=0.8)
    ax2.set_ylabel('Total Correction Magnitude', fontsize=12)
    ax2.set_title('Cumulative Correction Magnitude', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Max theta achieved
    ax3 = axes[2]
    means = [summary[e]['theta_max_mean'] for e in exp_labels]
    stds = [summary[e]['theta_max_std'] for e in exp_labels]
    ax3.bar(x, means, width, yerr=stds, capsize=3, color='forestgreen', alpha=0.8)
    ax3.set_ylabel('Max |θ| (rad)', fontsize=12)
    ax3.set_title('Maximum Pole Angle Achieved', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    summary_path = os.path.join(output_dir, 'ablation_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'\nSummary plot saved to: {summary_path}')


def plot_all_phase_plots(output_dir, experiments=None, seeds=None):
    '''Plot phase plots for all experiments from saved trajectory data.'''
    if experiments is None:
        experiments = EXPERIMENTS
    if seeds is None:
        seeds = SEEDS

    # Collect all trajectory data
    trajectories = []
    for exp_name in experiments:
        for seed in seeds:
            npz_path = os.path.join(output_dir, exp_name, f'seed_{seed}', 'trajectory_data.npz')
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                trajectories.append({
                    'experiment': exp_name,
                    'seed': seed,
                    'theta': data['theta'],
                    'theta_dot': data['theta_dot'],
                    'theta_uncert': data['theta_uncert'],
                    'theta_dot_uncert': data['theta_dot_uncert'],
                    'corrections': data['corrections'],
                    'theta_constraint': float(data['theta_constraint']),
                })

    if not trajectories:
        print('No trajectory data found. Run evaluation first.')
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
        ax.set_xlim(-0.25, 0.25)

    # Hide empty subplots
    for idx in range(n_exp, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'all_phase_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'All phase plots saved to: {plot_path}')
    return plot_path


def save_metrics_csv(all_metrics, output_dir):
    '''Save all metrics to a CSV file.'''
    import csv

    csv_path = os.path.join(output_dir, 'ablation_metrics.csv')

    if all_metrics:
        keys = all_metrics[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)

        print(f'Metrics saved to: {csv_path}')


def main():
    '''Main function to run all ablation experiments.'''
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate ablation experiments')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='Specific experiments to run (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Seeds to use (default: 42 62 821)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot from existing trajectory data (no evaluation)')
    args = parser.parse_args()

    # Filter experiments if specified
    experiments = args.experiments if args.experiments else EXPERIMENTS

    print('\n' + '#' * 60)
    print('# ADVERSARIAL REWARD ABLATION EVALUATION')
    print(f'# Experiments: {len(experiments)}')
    print(f'# Seeds: {args.seeds}')
    print(f'# Output: {OUTPUT_DIR}')
    print('#' * 60 + '\n')

    # Plot-only mode: just create phase plots from saved data
    if args.plot_only:
        plot_all_phase_plots(OUTPUT_DIR, experiments, args.seeds)
        print('\n' + '#' * 60)
        print('# PLOTTING COMPLETE')
        print('#' * 60 + '\n')
        return

    all_metrics = []

    for exp_name in experiments:
        for seed in args.seeds:
            model_dir = os.path.join(OUTPUT_DIR, exp_name, f'seed_{seed}')
            model_path = os.path.join(model_dir, 'model_latest.pt')

            if not os.path.exists(model_path):
                print(f'[SKIP] {exp_name} (seed={seed}) - model not found')
                continue

            try:
                metrics = evaluate_and_plot(exp_name, model_path, seed, model_dir)
                all_metrics.append(metrics)
            except Exception as e:
                print(f'[ERROR] {exp_name} (seed={seed}): {e}')

    # Create summary plots and save metrics
    if all_metrics:
        create_summary_plot(all_metrics, OUTPUT_DIR)
        save_metrics_csv(all_metrics, OUTPUT_DIR)
        plot_all_phase_plots(OUTPUT_DIR, experiments, args.seeds)

    print('\n' + '#' * 60)
    print('# EVALUATION COMPLETE')
    print(f'# Results saved to: {OUTPUT_DIR}')
    print('#' * 60 + '\n')


if __name__ == '__main__':
    main()
