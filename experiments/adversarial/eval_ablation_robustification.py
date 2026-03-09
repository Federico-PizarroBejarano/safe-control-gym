#!/usr/bin/env python3
"""Evaluate ablation × robustification sweep.

Runs evaluation on all trained models from train_ablation_robustification_sweep.sh
and generates comprehensive analysis of interaction effects.

Usage:
    python3 eval_ablation_robustification.py --seed 42
"""

import argparse
import csv
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
from robustification_metrics import compute_robustification_metrics

from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

# Import metrics function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def evaluate_ablation_robustification_sweep(seed, output_dir='./ablation_robustification', horizon_filter=None, append=False):
    """Evaluate all ablation × robustification sweep experiments.

    Args:
        seed: Random seed
        output_dir: Base directory containing trained models
        horizon_filter: optional set/list of horizons to evaluate (e.g., [20, 30])
        append: if True, append to existing metrics.csv instead of overwriting
    """
    seed_dir = os.path.join(output_dir, f'seed_{seed}')

    if not os.path.exists(seed_dir):
        print(f'[ERROR] Seed directory not found: {seed_dir}')
        return

    # Find all experiment directories
    exp_dirs = sorted([d for d in Path(seed_dir).iterdir() if d.is_dir()])

    # Optional horizon filter
    horizon_allow = None
    if horizon_filter:
        horizon_allow = set(int(h) for h in horizon_filter)

    print('=' * 80)
    print('ABLATION × ROBUSTIFICATION SWEEP EVALUATION')
    print(f'Seed: {seed}')
    print(f'Directory: {seed_dir}')
    print('=' * 80)
    print(f'\nFound {len(exp_dirs)} experiments\n')

    all_metrics = []

    for idx, exp_path in enumerate(exp_dirs, 1):
        exp_name = exp_path.name
        model_path = os.path.join(exp_path, 'model_latest.pt')

        if not os.path.exists(model_path):
            print(f'[{idx}/{len(exp_dirs)}] {exp_name} ... [SKIP] No model found')
            continue

        try:
            print(f'[{idx}/{len(exp_dirs)}] {exp_name} ...', end=' ', flush=True)

            # Clean up ACADOS-generated code to avoid dimension mismatches when horizon changes
            c_gen_path = os.path.join(os.path.dirname(__file__), 'c_generated_code')
            if os.path.exists(c_gen_path):
                shutil.rmtree(c_gen_path)

            # Parse experiment name to extract parameters
            parts = exp_name.split('_')

            # Find where robustification params start
            h_idx = next(i for i, p in enumerate(parts) if p.startswith('h'))
            ablation_name = '_'.join(parts[:h_idx])

            # Extract robustification parameters
            h_val = int(parts[h_idx][1:])  # h20 -> 20
            ch_val = int(parts[h_idx + 1][2:])  # ch5 -> 5
            w_val = float(parts[h_idx + 2][1:])  # w0.001 -> 0.001
            ts_val = parts[h_idx + 3][2:] == 'true'  # tstrue -> True

            # Skip if horizon filter is set and this horizon not included
            if horizon_allow and h_val not in horizon_allow:
                print(f'[{idx}/{len(exp_dirs)}] {exp_name} ... [SKIP] horizon {h_val} not in filter')
                continue

            # Setup configuration
            config_files = [
                'config_overrides/cartpole/ppo_cartpole.yaml',
                'config_overrides/cartpole/cartpole_track.yaml',
                'config_overrides/cartpole/nl_mpsc_cartpole.yaml',
            ]
            config_files = [os.path.join(os.path.dirname(__file__), f) for f in config_files]

            sys.argv = [
                '',
                '--task', 'cartpole',
                '--algo', 'ppo',
                '--safety_filter', 'nl_mpsc',
                '--overrides', *config_files,
                '--kv_overrides',
                f'sf_config.horizon={h_val}',
                f'sf_config.mpsc_cost_horizon={ch_val}',
                f'sf_config.max_w={w_val}',
                f'sf_config.use_terminal_set={ts_val}',
            ]

            fac = ConfigFactory()
            config = fac.merge()

            # Create environment
            env_func = partial(make, config.task, **config.task_config)
            env = env_func()

            # Load controller
            ctrl = make(config.algo, env_func, **config.algo_config)
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
            t0 = time.perf_counter()
            experiment = BaseExperiment(env, ctrl, safety_filter=safety_filter)
            cert_results, cert_metrics = experiment.run_evaluation(n_episodes=5)
            cert_elapsed = time.perf_counter() - t0
            cert_steps = sum(len(o) for o in cert_results['obs'])

            # Run uncertified evaluation
            ctrl.reset()
            t1 = time.perf_counter()
            experiment_uncert = BaseExperiment(env, ctrl)
            uncert_results, uncert_metrics = experiment_uncert.run_evaluation(n_episodes=5)
            uncert_elapsed = time.perf_counter() - t1

            ctrl.close()
            safety_filter.close()

            # Compute robustification metrics
            mpsc_results = cert_results['safety_filter_data']
            metrics = compute_robustification_metrics(
                cert_results,
                uncert_results,
                mpsc_results,
                config,
                info={
                    'cert_compute_time_total': cert_elapsed,
                    'cert_compute_time_per_step': cert_elapsed / max(cert_steps, 1),
                    'uncert_compute_time_total': uncert_elapsed,
                    'uncert_compute_time_per_step': uncert_elapsed / max(sum(len(o) for o in uncert_results['obs']), 1),
                },
            )

            # Add experiment metadata
            metrics['experiment'] = exp_name
            metrics['ablation'] = ablation_name
            metrics['seed'] = seed
            metrics['horizon'] = h_val
            metrics['cost_horizon'] = ch_val
            metrics['max_w'] = w_val
            metrics['terminal_set'] = ts_val

            all_metrics.append(metrics)
            print('✓')

        except Exception as e:
            print(f'[ERROR] {type(e).__name__}: {str(e)[:80]}')
            # Don't stop - continue with remaining experiments
            continue

    # Save results
    if all_metrics:
        csv_path = os.path.join(seed_dir, 'metrics.csv')
        print(f'\nSaving {len(all_metrics)} metric rows to {csv_path} (append={append})...')

        keys = list(all_metrics[0].keys())
        write_header = True
        mode = 'w'
        if append and os.path.exists(csv_path):
            mode = 'a'
            write_header = False

        with open(csv_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerows(all_metrics)

        print(f'Metrics saved to: {csv_path}')

        # Print summary statistics
        df = pd.DataFrame(all_metrics)
        print('\n' + '=' * 80)
        print('SUMMARY STATISTICS')
        print('=' * 80)
        print('\nViolation rates by ablation:')
        print(df.groupby('ablation')[['cert_violation_rate']].agg(['mean', 'std', 'min', 'max']))
        print('\nViolation rates by horizon:')
        print(df.groupby('horizon')[['cert_violation_rate']].agg(['mean', 'std', 'min', 'max']))
        print('\nViolation rates by max_w:')
        print(df.groupby('max_w')[['cert_violation_rate']].agg(['mean', 'std', 'min', 'max']))
        print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate ablation × robustification sweep')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', default='./ablation_robustification',
                        help='Base output directory')
    parser.add_argument('--horizons', type=str, default='',
                        help='Comma-separated list of horizons to evaluate (e.g., "20,30")')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing metrics.csv instead of overwriting')
    args = parser.parse_args()
    horizons = [int(h) for h in args.horizons.split(',') if h.strip()] if args.horizons else None

    evaluate_ablation_robustification_sweep(
        args.seed,
        args.output_dir,
        horizon_filter=horizons,
        append=args.append,
    )


if __name__ == '__main__':
    main()
