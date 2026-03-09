#!/usr/bin/env python3
"""Utility for visualizing PPO adversarial training logs.

Reads scalar log files produced by ``ExperimentLogger`` (saved under
``<output_dir>/logs``) and generates summary plots for reward shaping,
policy statistics, and constraint behaviour.

Example:
    python analyze_training.py \
        --log-dir models/rl_models/cartpole/track/ppo/temp \
        --output-dir figures/adv_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

MetricSeries = Tuple[np.ndarray, np.ndarray]

# Mapping from friendly panel title to list of metric names to plot
PLOT_GROUPS: Dict[str, List[str]] = {
    'Adversarial Reward': [
        'adv_reward/adv_reward_raw_mean',
        'adv_reward/adv_reward_scaled_mean',
        'adv_reward/adv_reward_scaled_max',
        'adv_reward/adv_reward_scaled_min',
    ],
    'Policy Optimization': [
        'loss/approx_kl',
        'loss/policy_loss',
        'loss/value_loss',
        'loss/entropy_loss',
    ],
    'Training Performance': [
        'stat/ep_constraint_violation',
        'stat/ep_length',
        'stat/ep_return',
    ],
    'Evaluation Summary': [
        'stat_eval/constraint_violation',
        'stat_eval/ep_return',
        'stat_eval/ep_length',
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot PPO adversarial training logs.')
    parser.add_argument(
        '--log-dir',
        required=True,
        help="Directory containing ExperimentLogger outputs (expects a 'logs' sub-folder).",
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help="Where to save generated figures (default: '<log-dir>/plots').",
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=120,
        help='Dots-per-inch for saved figures (default: 120).',
    )
    return parser.parse_args()


def load_metric(log_root: Path, name: str) -> MetricSeries | None:
    """Load a metric CSV exported by :class:`FileLogger`.

    Args:
        log_root: Directory containing the ``logs`` folder.
        name: Metric name (slashes map to subdirectories).

    Returns:
        Tuple of (steps, values) arrays or ``None`` if the log does not exist.
    """

    log_path = log_root / 'logs' / f'{name}.log'
    if not log_path.exists():
        return None

    steps: List[float] = []
    values: List[float] = []
    with log_path.open('r', encoding='utf-8') as f:
        # Skip header line: "step,value"
        next(f, None)
        for line in f:
            step_str, val_str = line.strip().split(',')
            steps.append(float(step_str))
            values.append(float(val_str))
    if not steps:
        return None
    return np.asarray(steps), np.asarray(values)


def ensure_output_dir(path: Path | None, log_dir: Path) -> Path:
    if path is None:
        path = log_dir / 'plots'
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_group(
    axes: Iterable[plt.Axes],
    log_dir: Path,
) -> None:
    for ax, (title, metrics) in zip(axes, PLOT_GROUPS.items()):
        for metric in metrics:
            series = load_metric(log_dir, metric)
            if series is None:
                continue
            steps, values = series
            ax.plot(steps, values, label=metric)
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.legend(fontsize='small')


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    if not (log_dir / 'logs').exists():
        raise FileNotFoundError(f"Could not find 'logs' folder under {log_dir}")

    output_dir = ensure_output_dir(Path(args.output_dir).expanduser().resolve() if args.output_dir else None, log_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=args.dpi, constrained_layout=True)
    axes_flat = axes.flatten()
    plot_group(axes_flat, log_dir)

    figure_path = output_dir / 'training_summary.png'
    fig.suptitle(f'PPO Adversarial Training Summary\n{log_dir}')
    fig.savefig(figure_path, dpi=args.dpi)
    print(f'Saved summary figure to {figure_path}')

    # Optional second figure focusing on adversarial reward distribution
    reward_metrics = [
        'adv_reward/adv_reward_raw_mean',
        'adv_reward/adv_reward_raw_max',
        'adv_reward/adv_reward_raw_min',
    ]
    data = [load_metric(log_dir, metric) for metric in reward_metrics]
    if any(series is not None for series in data):
        plt.figure(figsize=(10, 5), dpi=args.dpi)
        for metric, series in zip(reward_metrics, data):
            if series is None:
                continue
            steps, values = series
            plt.plot(steps, values, label=metric)
        plt.title('Adversarial Reward (Raw)')
        plt.xlabel('Step')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.legend(fontsize='small')
        raw_path = output_dir / 'adv_reward_raw.png'
        plt.savefig(raw_path, dpi=args.dpi)
        print(f'Saved raw reward figure to {raw_path}')


if __name__ == '__main__':
    main()
