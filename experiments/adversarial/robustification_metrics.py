#!/usr/bin/env python3
"""Updated run_ablation.py with robustification metrics for cartpole.

This version extends run_ablation.py to:
1. Compute robustification-specific metrics
2. Handle robustification sweep experiments
3. Generate summary tables and plots
"""

# This is a companion/extension to the existing run_ablation.py
# Add these metric computation functions to run_ablation.py or use separately

import numpy as np

# State index for primary angle constraint (theta) per plant family.
THETA_IDX_BY_SYSTEM = {
    'cartpole': 2,
    'quadrotor_2D': 4,
    'quadrotor': 4,
}


def compute_robustification_metrics(cert_results, uncert_results, mpsc_results, config, system='cartpole', info=None):
    """Compute robustification metrics for violation, control, and performance analysis.

    Focuses on the requested metrics:
        - Violation rate (episode-level): fraction of episodes with any violation
        - Time-to-first-violation (steps until first violation; T if none)
        - Severity: integrated negative slack (sum of violation magnitudes)
        - Corrections: magnitude stats and rate
        - Rate of change of inputs
        - Control effort (L1/L2)
        - Compute time (provided via `info` dict)
    """
    metrics = {}

    # Results are stored per episode; convert to lists for convenience
    cert_obs_list = cert_results['obs']
    uncert_obs_list = uncert_results['obs']
    cert_actions_list = cert_results['action']
    uncert_actions_list = uncert_results['action']
    corrections_list = mpsc_results.get('correction', [])

    # Get constraint bounds
    try:
        constraints = config.task_config.constraints[0]
        upper_bounds = constraints.upper_bounds
        lower_bounds = constraints.lower_bounds
    except Exception:
        # Defaults for cartpole
        upper_bounds = [2.4, 100, 0.2, 100]
        lower_bounds = [-2.4, -100, -0.2, -100]

    theta_idx = THETA_IDX_BY_SYSTEM.get(system, 2)
    if theta_idx >= len(upper_bounds):
        raise ValueError(f'constraint bounds too short for system={system!r} theta_idx={theta_idx}')
    theta_ub = upper_bounds[theta_idx]
    theta_lb = lower_bounds[theta_idx]

    # Helpers to flatten episode lists when needed
    def concat(list_arrs):
        return np.concatenate(list_arrs, axis=0) if len(list_arrs) > 1 else list_arrs[0]

    # ===== VIOLATION METRICS =====
    # Episode-level: fraction of episodes with any violation
    def episode_violation_flags(obs_list):
        flags = []
        for obs in obs_list:
            vio = np.logical_or(obs[:, theta_idx] > theta_ub, obs[:, theta_idx] < theta_lb)
            flags.append(np.any(vio))
        return np.array(flags)

    cert_vio_flags = episode_violation_flags(cert_obs_list)
    uncert_vio_flags = episode_violation_flags(uncert_obs_list)
    metrics['cert_episode_violation_rate'] = float(np.mean(cert_vio_flags))  # fraction of episodes
    metrics['uncert_episode_violation_rate'] = float(np.mean(uncert_vio_flags))

    # Timestep-level: fraction of timesteps with violation (more granular!)
    def timestep_violation_rate(obs_list):
        all_violations = []
        for obs in obs_list:
            vio = np.logical_or(obs[:, theta_idx] > theta_ub, obs[:, theta_idx] < theta_lb)
            all_violations.append(vio)
        all_vio = np.concatenate(all_violations)
        return float(np.mean(all_vio))

    metrics['cert_violation_rate'] = timestep_violation_rate(cert_obs_list)  # timestep-level
    metrics['uncert_violation_rate'] = timestep_violation_rate(uncert_obs_list)

    # Time to first violation (across episodes)
    def time_to_first_violation(obs_list):
        ttfv = []
        for obs in obs_list:
            vio = np.logical_or(obs[:, theta_idx] > theta_ub, obs[:, theta_idx] < theta_lb)
            idx = np.where(vio)[0]
            ttfv.append(int(idx[0]) if len(idx) else len(obs))
        return ttfv

    metrics['cert_time_to_first_violation'] = min(time_to_first_violation(cert_obs_list)) if cert_obs_list else 0
    metrics['uncert_time_to_first_violation'] = min(time_to_first_violation(uncert_obs_list)) if uncert_obs_list else 0

    # Flatten for severity/action-based metrics
    cert_obs = concat(cert_obs_list)
    uncert_obs = concat(uncert_obs_list)
    cert_actions = concat(cert_actions_list)
    uncert_actions = concat(uncert_actions_list)

    corrections = np.atleast_2d(concat([np.atleast_1d(c) for c in corrections_list])) if len(corrections_list) > 0 else np.zeros((len(cert_actions), 1))
    if corrections.shape[0] == 1:
        corrections = corrections.T

    T = len(cert_obs)

    # ===== SEVERITY METRICS (integrated negative slack) =====
    # Violation magnitude is how far outside bounds we are
    cert_violation_mag = np.maximum(0, cert_obs[:, theta_idx] - theta_ub) + np.maximum(0, theta_lb - cert_obs[:, theta_idx])
    uncert_violation_mag = np.maximum(0, uncert_obs[:, theta_idx] - theta_ub) + np.maximum(0, theta_lb - uncert_obs[:, theta_idx])

    metrics['cert_integrated_slack'] = float(np.sum(cert_violation_mag))
    metrics['uncert_integrated_slack'] = float(np.sum(uncert_violation_mag))
    metrics['cert_max_violation'] = float(np.max(cert_violation_mag))
    metrics['uncert_max_violation'] = float(np.max(uncert_violation_mag))

    # ===== CORRECTION METRICS =====
    correction_magnitude = np.linalg.norm(corrections, axis=1)
    has_correction = correction_magnitude > 1e-6

    metrics['num_corrections'] = int(np.sum(has_correction))
    metrics['total_correction_magnitude'] = float(np.sum(correction_magnitude))
    metrics['max_correction'] = float(np.max(correction_magnitude)) if correction_magnitude.size else 0.0
    metrics['mean_correction'] = float(np.mean(correction_magnitude[has_correction]) if np.any(has_correction) else 0)
    metrics['correction_rate'] = float(np.sum(has_correction) / max(T, 1))

    # ===== INPUT RATE OF CHANGE =====
    cert_action_deltas = np.diff(cert_actions, axis=0) if len(cert_actions) > 1 else np.zeros_like(cert_actions)
    uncert_action_deltas = np.diff(uncert_actions, axis=0) if len(uncert_actions) > 1 else np.zeros_like(uncert_actions)

    def delta_stats(deltas):
        if len(deltas) == 0:
            return 0.0, 0.0
        norms = np.linalg.norm(deltas, axis=1)
        return float(np.mean(norms)), float(np.max(norms))

    metrics['cert_mean_action_rate'], metrics['cert_max_action_rate'] = delta_stats(cert_action_deltas)
    metrics['uncert_mean_action_rate'], metrics['uncert_max_action_rate'] = delta_stats(uncert_action_deltas)

    # ===== CONTROL EFFORT =====
    metrics['cert_control_effort_l1'] = float(np.sum(np.abs(cert_actions)))
    metrics['uncert_control_effort_l1'] = float(np.sum(np.abs(uncert_actions)))
    metrics['cert_control_effort_l2'] = float(np.sum(cert_actions ** 2))
    metrics['uncert_control_effort_l2'] = float(np.sum(uncert_actions ** 2))

    # ===== COMPUTE TIME (if provided) =====
    if info is not None:
        metrics['cert_compute_time_total'] = info.get('cert_compute_time_total', None)
        metrics['cert_compute_time_per_step'] = info.get('cert_compute_time_per_step', None)
        metrics['uncert_compute_time_total'] = info.get('uncert_compute_time_total', None)
        metrics['uncert_compute_time_per_step'] = info.get('uncert_compute_time_per_step', None)

    return metrics


def format_robustification_summary(all_metrics):
    """Create formatted summary table of robustification experiments.

    Args:
        all_metrics: List of metric dicts

    Returns:
        str: Formatted summary table
    """
    if not all_metrics:
        return 'No metrics to summarize'

    # Sort by robustification parameters if available
    sorted_metrics = sorted(all_metrics, key=lambda m: (
        m.get('horizon', 0),
        m.get('cost_horizon', 0),
        m.get('max_w', 0),
        m.get('terminal_set', False)
    ))

    lines = []
    lines.append('\n' + '=' * 120)
    lines.append('ROBUSTIFICATION STUDY SUMMARY - CARTPOLE')
    lines.append('=' * 120)

    # Header
    header = (f"{'Exp':<25} {'H':<4} {'CH':<3} {'W':<8} {'TS':<3} "
              f"{'CViol':<7} {'UViol':<7} {'CSlack':<8} {'USlack':<8} "
              f"{'TtFV':<5} {'#Corr':<6} {'MaxC':<6} {'CtrlEf':<7}")
    lines.append(header)
    lines.append('-' * 120)

    for m in sorted_metrics:
        exp_name = m.get('experiment', 'unknown')[:24]
        h = m.get('horizon', '-')
        ch = m.get('cost_horizon', '-')
        w = f"{m.get('max_w', 0):.4f}" if isinstance(m.get('max_w', 0), float) else '-'
        ts = 'Y' if m.get('terminal_set', False) else 'N'

        cviol = f"{m.get('cert_violation_rate', 0)*100:.1f}%"
        uviol = f"{m.get('uncert_violation_rate', 0)*100:.1f}%"
        cslack = f"{m.get('cert_integrated_slack', 0):.2f}"
        uslack = f"{m.get('uncert_integrated_slack', 0):.2f}"
        ttfv = m.get('uncert_time_to_first_violation', 999)
        nc = m.get('num_corrections', 0)
        maxc = f"{m.get('max_correction', 0):.3f}"
        effort = f"{m.get('cert_control_effort_l1', 0):.1f}"

        line = (f'{exp_name:<25} {h:<4} {ch:<3} {w:<8} {ts:<3} '
                f'{cviol:<7} {uviol:<7} {cslack:<8} {uslack:<8} '
                f'{ttfv:<5} {nc:<6} {maxc:<6} {effort:<7}')
        lines.append(line)

    lines.append('=' * 120)
    lines.append('\nLegend: H=Horizon, CH=Cost Horizon, W=Max Constraint Tightening (max_w), TS=Terminal Set')
    lines.append('        CViol=Certified Violation Rate, UViol=Uncertified Violation Rate')
    lines.append('        CSlack=Certified Slack, USlack=Uncertified Slack, TtFV=Time to First Violation')
    lines.append('        #Corr=Number of Corrections, MaxC=Max Correction, CtrlEf=Control Effort (L1)')

    return '\n'.join(lines)
