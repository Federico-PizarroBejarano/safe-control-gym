#!/usr/bin/env python3
"""Visualize ablation × robustification sweep results.

Focused visualization of key robustification metrics:
- Violation rate, time-to-first-violation, severity
- Correction magnitude, control effort, smoothness
- Compute time, RMSE

Usage:
    python3 plot_ablation_robustification.py --seed 42
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_ablation_robustification(seed, output_dir=None):
    """Create comprehensive visualizations of ablation × robustification interactions.

    Args:
        seed: Random seed
        output_dir: Base directory containing results (default: script's directory)
    """
    # If no output_dir provided, use the directory of this script
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    # Support both:
    # 1) output_dir = .../experiments/adversarial
    # 2) output_dir = .../experiments/adversarial/ablation_robustification
    if os.path.basename(os.path.normpath(output_dir)) == 'ablation_robustification':
        base_dir = output_dir
    else:
        base_dir = os.path.join(output_dir, 'ablation_robustification')

    seed_dir = os.path.join(base_dir, f'seed_{seed}')
    csv_path = os.path.join(seed_dir, 'metrics.csv')
    plot_dir = os.path.join(seed_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f'[ERROR] Metrics CSV not found: {csv_path}')
        print(f'Available path: {os.path.dirname(csv_path)}')
        print(f'Run: python3 eval_ablation_robustification.py --seed {seed}')
        return

    df = pd.read_csv(csv_path)

    print(f'Loaded {len(df)} experiments from {csv_path}')
    print(f'Ablations: {df["ablation"].nunique()} variants')
    print(f'Horizons: {sorted(df["horizon"].unique())}')
    print(f'Max_w: {sorted(df["max_w"].unique())}')
    print(f'Cost Horizons: {sorted(df["cost_horizon"].unique())}')

    # Create output directory
    os.makedirs(plot_dir, exist_ok=True)

    # ========================================================================
    # PLOT 1: KEY METRICS SUMMARY - Bar charts comparing ablations
    # ========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    metrics = [
        ('cert_violation_rate', 'Violation Rate', 'Fraction'),
        ('cert_time_to_first_violation', 'Time to 1st Violation', 'Steps'),
        ('cert_integrated_slack', 'Severity (Slack)', 'Sum'),
        ('max_correction', 'Max Correction', 'Magnitude'),
        ('cert_mean_action_rate', 'Action Rate (Δu)', 'Rate'),
        ('cert_control_effort_l1', 'Control Effort (L1)', 'Total'),
        ('cert_compute_time_per_step', 'Compute Time/Step', 'Seconds'),
        ('rmse_cert', 'RMSE (Certified)', 'Error'),
    ]

    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 4, idx % 4]

        # Check if column exists
        if metric not in df.columns:
            # Handle missing columns gracefully
            if metric == 'rmse_cert':
                # Try alternative name
                metric = 'cert_mean_action_rate'  # Placeholder
                title = f'{title} (N/A)'

        # Aggregate by ablation
        agg = df.groupby('ablation')[metric].agg(['mean', 'std']).sort_values('mean')

        # Color code: green = low (good for violations), red = high
        if 'violation' in metric or 'slack' in metric or 'effort' in metric or 'time' in metric:
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(agg)))
        else:
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(agg)))

        ax.barh(range(len(agg)), agg['mean'], xerr=agg['std'],
                capsize=3, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_yticks(range(len(agg)))
        ax.set_yticklabels(agg.index, fontsize=9)
        ax.set_xlabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels on bars
        for i, (mean_val, std_val) in enumerate(zip(agg['mean'], agg['std'])):
            if metric in ['cert_violation_rate', 'correction_rate']:
                label = f'{mean_val:.1%}'
            elif metric in ['cert_compute_time_per_step', 'cert_compute_time_total']:
                label = f'{mean_val:.4f}'
            elif 'time' in metric and 'first' in metric:
                label = f'{int(mean_val)}'
            else:
                label = f'{mean_val:.1f}'
            ax.text(mean_val, i, f'  {label}', va='center', fontsize=8)

    plt.suptitle(f'Robustification Metrics Summary (Seed {seed})', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'summary_metrics.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ Saved: summary_metrics.png')

    # ========================================================================
    # PLOT 2: BEST PERFORMERS - Highest violation rates (most adversarial)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Best ablations (highest violations = most adversarial)
    ax = axes[0]
    best_abl = df.groupby('ablation')['cert_violation_rate'].agg(['mean', 'max', 'std']).sort_values('mean', ascending=False).head(10)

    colors_best = plt.cm.Greens(np.linspace(0.5, 0.9, len(best_abl)))
    ax.barh(range(len(best_abl)), best_abl['mean'], xerr=best_abl['std'],
            capsize=3, color=colors_best, alpha=0.85, edgecolor='black', linewidth=0.7)
    ax.set_yticks(range(len(best_abl)))
    ax.set_yticklabels(best_abl.index, fontsize=10)
    ax.set_xlabel('Mean Violation Rate', fontsize=11)
    ax.set_title('Best Ablations (Most Adversarial - Highest Violations)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, (mean_val, max_val) in enumerate(zip(best_abl['mean'], best_abl['max'])):
        ax.text(mean_val, i, f'  {mean_val:.1%} (max: {max_val:.1%})', va='center', fontsize=9)

    # Best individual experiments (most adversarial)
    ax = axes[1]
    # De-duplicate by full parameter combination to avoid showing same experiment multiple times
    df_dedup = df.drop_duplicates(subset=['ablation', 'horizon', 'cost_horizon', 'max_w', 'terminal_set'], keep='first')
    best_exp = df_dedup.nlargest(15, 'cert_violation_rate')[['experiment', 'ablation', 'horizon', 'cost_horizon', 'max_w', 'cert_violation_rate', 'cert_integrated_slack']]

    labels = [f"{row['ablation'][:8]}_h{int(row['horizon'])}_ch{int(row['cost_horizon'])}_w{row['max_w']:.3f}"
              for _, row in best_exp.iterrows()]
    colors_exp = plt.cm.Greens(np.linspace(0.5, 0.9, len(best_exp)))

    ax.barh(range(len(best_exp)), best_exp['cert_violation_rate'].values,
            color=colors_exp, alpha=0.85, edgecolor='black', linewidth=0.7)
    ax.set_yticks(range(len(best_exp)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Violation Rate', fontsize=11)
    ax.set_title('Best 15 Experiments (Most Adversarial)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, val in enumerate(best_exp['cert_violation_rate'].values):
        ax.text(val, i, f'  {val:.1%}', va='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'best_performers.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ Saved: best_performers.png')

    # ========================================================================
    # PLOT 3: WORST PERFORMERS - Lowest violation rates (least adversarial)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Worst ablations (lowest violations = least adversarial)
    ax = axes[0]
    worst_abl = df.groupby('ablation')['cert_violation_rate'].agg(['mean', 'min', 'std']).sort_values('mean').head(10)

    colors_worst = plt.cm.Reds(np.linspace(0.5, 0.9, len(worst_abl)))
    ax.barh(range(len(worst_abl)), worst_abl['mean'], xerr=worst_abl['std'],
            capsize=3, color=colors_worst, alpha=0.85, edgecolor='black', linewidth=0.7)
    ax.set_yticks(range(len(worst_abl)))
    ax.set_yticklabels(worst_abl.index, fontsize=10)
    ax.set_xlabel('Mean Violation Rate', fontsize=11)
    ax.set_title('Worst Ablations (Least Adversarial - Lowest Violations)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, (mean_val, min_val) in enumerate(zip(worst_abl['mean'], worst_abl['min'])):
        ax.text(mean_val, i, f'  {mean_val:.1%} (min: {min_val:.1%})', va='center', fontsize=9)

    # Worst individual experiments (least adversarial)
    ax = axes[1]
    # De-duplicate by full parameter combination to avoid showing same experiment multiple times
    worst_exp = df_dedup.nsmallest(15, 'cert_violation_rate')[['experiment', 'ablation', 'horizon', 'cost_horizon', 'max_w', 'cert_violation_rate', 'num_corrections']]

    labels = [f"{row['ablation'][:8]}_h{int(row['horizon'])}_ch{int(row['cost_horizon'])}_w{row['max_w']:.3f}"
              for _, row in worst_exp.iterrows()]
    colors_exp = plt.cm.Reds(np.linspace(0.5, 0.9, len(worst_exp)))

    ax.barh(range(len(worst_exp)), worst_exp['cert_violation_rate'].values,
            color=colors_exp, alpha=0.85, edgecolor='black', linewidth=0.7)
    ax.set_yticks(range(len(worst_exp)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Violation Rate', fontsize=11)
    ax.set_title('Worst 15 Experiments (Least Adversarial)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, val in enumerate(worst_exp['cert_violation_rate'].values):
        ax.text(val, i, f'  {val:.1%}', va='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'worst_performers.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ Saved: worst_performers.png')

    # ========================================================================
    # PLOT 4: INTERACTION EFFECTS - Ablation × MPC Parameters
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Violation rate vs max_w by ablation
    ax = axes[0, 0]
    for ablation in sorted(df['ablation'].unique()):
        subset = df[df['ablation'] == ablation]
        agg = subset.groupby('max_w')['cert_violation_rate'].mean()
        ax.plot(agg.index, agg.values, marker='o', label=ablation, linewidth=2.5, markersize=9)
    ax.set_xscale('log')
    ax.set_xlabel('Max Disturbance (max_w)', fontsize=11)
    ax.set_ylabel('Violation Rate', fontsize=11)
    ax.set_title('Safety vs Disturbance Bound', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    # Violation rate vs horizon by ablation (or cost_horizon if only one horizon)
    ax = axes[0, 1]
    num_horizons = len(df['horizon'].unique())

    if num_horizons > 1:
        # Multiple horizons: plot line
        for ablation in sorted(df['ablation'].unique()):
            subset = df[df['ablation'] == ablation]
            agg = subset.groupby('horizon')['cert_violation_rate'].mean()
            ax.plot(agg.index, agg.values, marker='s', label=ablation, linewidth=2.5, markersize=9)
        ax.set_xlabel('MPC Horizon', fontsize=11)
        ax.set_ylabel('Violation Rate', fontsize=11)
        ax.set_title('Safety vs Planning Horizon', fontsize=12, fontweight='bold')
    else:
        # Single horizon: show cost_horizon instead
        for ablation in sorted(df['ablation'].unique()):
            subset = df[df['ablation'] == ablation]
            agg = subset.groupby('cost_horizon')['cert_violation_rate'].mean()
            ax.plot(agg.index, agg.values, marker='s', label=ablation, linewidth=2.5, markersize=9)
        ax.set_xlabel('Cost Horizon', fontsize=11)
        ax.set_ylabel('Violation Rate', fontsize=11)
        ax.set_title('Safety vs Cost Horizon (Single MPC Horizon)', fontsize=12, fontweight='bold')

    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    # Corrections vs max_w by ablation
    ax = axes[1, 0]
    for ablation in sorted(df['ablation'].unique()):
        subset = df[df['ablation'] == ablation]
        agg = subset.groupby('max_w')['num_corrections'].mean()
        ax.plot(agg.index, agg.values, marker='o', label=ablation, linewidth=2.5, markersize=9)
    ax.set_xscale('log')
    ax.set_xlabel('Max Disturbance (max_w)', fontsize=11)
    ax.set_ylabel('Number of Corrections', fontsize=11)
    ax.set_title('Filter Workload vs Disturbance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    # Severity vs control effort (trade-off)
    ax = axes[1, 1]
    for ablation in sorted(df['ablation'].unique()):
        subset = df[df['ablation'] == ablation]
        ax.scatter(subset['cert_control_effort_l1'], subset['cert_integrated_slack'],
                   label=ablation, s=80, alpha=0.7)
    ax.set_xlabel('Control Effort (L1)', fontsize=11)
    ax.set_ylabel('Severity (Integrated Slack)', fontsize=11)
    ax.set_title('Trade-off: Control Effort vs Severity', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'interaction_effects.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ Saved: interaction_effects.png')

    # ========================================================================
    # PLOT 5: HEATMAPS - Key metrics across ablation × parameter grid
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    heatmap_metrics = [
        ('cert_violation_rate', 'Violation Rate', 0, 1),
        ('cert_integrated_slack', 'Severity (Slack)', 0, None),
        ('num_corrections', 'Corrections', 0, None),
        ('max_correction', 'Max Correction', 0, None),
        ('cert_control_effort_l1', 'Control Effort', 0, None),
        ('cert_mean_action_rate', 'Action Rate (Δu)', 0, None),
    ]

    for idx, (metric, title, vmin, vmax) in enumerate(heatmap_metrics):
        ax = axes[idx // 3, idx % 3]

        # Pivot: ablation × max_w
        pivot = df.pivot_table(
            index='ablation',
            columns='max_w',
            values=metric,
            aggfunc='mean'
        )
        pivot = pivot.loc[sorted(pivot.index)]

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
                    cbar_kws={'label': metric}, vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='gray')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Max Disturbance (max_w)', fontsize=10)
        ax.set_ylabel('Ablation Variant', fontsize=10)

    plt.suptitle('Metric Heatmaps: Ablation × max_w', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'heatmaps.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('✓ Saved: heatmaps.png')

    print(f'\nAll plots saved to: {plot_dir}')

    # ========================================================================
    # SUMMARY TABLE - Print key findings
    # ========================================================================
    print('\n' + '=' * 80)
    print('KEY FINDINGS SUMMARY')
    print('=' * 80)

    # Overall statistics
    print('\nOverall Statistics:')
    print(f'  Total experiments: {len(df)}')
    print(f'  Mean violation rate: {df["cert_violation_rate"].mean():.2%}')
    print(f'  Std violation rate: {df["cert_violation_rate"].std():.2%}')
    print(f'  Min violation rate: {df["cert_violation_rate"].min():.2%}')
    print(f'  Max violation rate: {df["cert_violation_rate"].max():.2%}')

    # Best ablation (most adversarial - highest violations)
    best_ablation = df.groupby('ablation')['cert_violation_rate'].mean().idxmax()
    best_viol = df.groupby('ablation')['cert_violation_rate'].mean().max()
    print('\n Best Ablation (Most Adversarial): ' + str(best_ablation))
    print(f'   Violation rate: {best_viol:.2%}')

    # Worst ablation (least adversarial - lowest violations)
    worst_ablation = df.groupby('ablation')['cert_violation_rate'].mean().idxmin()
    worst_viol = df.groupby('ablation')['cert_violation_rate'].mean().min()
    print('\n Worst Ablation (Least Adversarial): ' + str(worst_ablation))
    print(f'   Violation rate: {worst_viol:.2%}')

    # Best MPC parameter (for adversarial agent - highest violations)
    if len(df['horizon'].unique()) > 1:
        best_horizon = df.groupby('horizon')['cert_violation_rate'].mean().idxmax()
        best_h_viol = df.groupby('horizon')['cert_violation_rate'].mean().max()
        print(f'\nMost Adversarial Horizon: {best_horizon} (violation rate: {best_h_viol:.2%})')

    best_maxw = df.groupby('max_w')['cert_violation_rate'].mean().idxmax()
    best_w_viol = df.groupby('max_w')['cert_violation_rate'].mean().max()
    print(f'Most Adversarial max_w: {best_maxw} (violation rate: {best_w_viol:.2%})')

    # Best overall combination (most adversarial - highest violations)
    best_idx = df['cert_violation_rate'].idxmax()
    best_row = df.loc[best_idx]
    print('\n Best Overall Combination (Most Adversarial):')
    print(f'   Experiment: {best_row["experiment"]}')
    print(f'   Ablation: {best_row["ablation"]}')
    print(f'   Horizon: {int(best_row["horizon"])}, Cost Horizon: {int(best_row["cost_horizon"])}')
    print(f'   Max_w: {best_row["max_w"]}, Terminal Set: {best_row["terminal_set"]}')
    print(f'   Violation Rate: {best_row["cert_violation_rate"]:.2%}')
    print(f'   Time to 1st Violation: {int(best_row["cert_time_to_first_violation"])} steps')
    print(f'   Severity (Slack): {best_row["cert_integrated_slack"]:.2f}')
    print(f'   Corrections: {int(best_row["num_corrections"])} ({best_row["correction_rate"]:.1%} rate)')
    print(f'   Max Correction: {best_row["max_correction"]:.2f}')
    print(f'   Control Effort: {best_row["cert_control_effort_l1"]:.1f}')
    print(f'   Compute Time/Step: {best_row["cert_compute_time_per_step"]:.5f}s')

    # Worst overall combination (least adversarial - lowest violations)
    worst_idx = df['cert_violation_rate'].idxmin()
    worst_row = df.loc[worst_idx]
    print('\n Worst Overall Combination (Least Adversarial):')
    print(f'   Experiment: {worst_row["experiment"]}')
    print(f'   Ablation: {worst_row["ablation"]}')
    print(f'   Horizon: {int(worst_row["horizon"])}, Cost Horizon: {int(worst_row["cost_horizon"])}')
    print(f'   Max_w: {worst_row["max_w"]}, Terminal Set: {worst_row["terminal_set"]}')
    print(f'   Violation Rate: {worst_row["cert_violation_rate"]:.2%}')
    print(f'   Severity (Slack): {worst_row["cert_integrated_slack"]:.2f}')

    # Comparison table (sorted by descending violation rate for adversarial objective)
    print('\n' + '-' * 80)
    print('Ablation Rankings (by mean violation rate - higher = more adversarial):')
    print('-' * 80)
    rankings = df.groupby('ablation').agg({
        'cert_violation_rate': ['mean', 'std', 'min', 'max'],
        'cert_integrated_slack': 'mean',
        'num_corrections': 'mean',
        'cert_control_effort_l1': 'mean',
    }).round(3)
    rankings.columns = ['_'.join(col).strip() for col in rankings.columns.values]
    rankings = rankings.sort_values('cert_violation_rate_mean', ascending=False)  # Descending for adversarial

    for idx, (ablation, row) in enumerate(rankings.iterrows(), 1):
        print(f'{idx:2d}. {ablation:20s} | Viol: {row["cert_violation_rate_mean"]:.1%} ± {row["cert_violation_rate_std"]:.1%} | '
              f'Slack: {row["cert_integrated_slack_mean"]:6.2f} | Corr: {row["num_corrections_mean"]:5.1f} | '
              f'Effort: {row["cert_control_effort_l1_mean"]:6.1f}')

    print('=' * 80)


def main():
    parser = argparse.ArgumentParser(description='Plot ablation × robustification results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', default=None,
                        help='Base output directory (adversarial dir or ablation_robustification dir)')
    args = parser.parse_args()

    plot_ablation_robustification(args.seed, args.output_dir)


if __name__ == '__main__':
    main()
