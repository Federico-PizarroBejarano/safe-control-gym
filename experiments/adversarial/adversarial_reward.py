'''Controller-only adversarial reward: encourage NL-MPSC corrections without plant disturbances.'''

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

import numpy as np


@dataclass
class AdversarialNominalConfig:
    '''Hyperparameters for nominal-command adversarial reward (softplus on raw sum).'''

    temperature: float = 15.0
    use_correction_reward: bool = True
    use_correction_ratio: bool = True
    use_correction_bonus: bool = True
    use_no_correction_penalty: bool = True
    use_theta_reward: bool = True
    use_velocity_reward: bool = True
    use_oscillation_reward: bool = True
    use_cart_penalty: bool = True
    use_stability_penalty: bool = True
    use_altitude_penalty: bool = True
    use_position_reward: bool = True
    w_correction: float = 30.0
    w_correction_ratio: float = 40.0
    w_theta: float = 5.0
    w_velocity: float = 3.0
    w_oscillation: float = 2.0
    w_cart_penalty: float = 2.0
    w_termination_penalty: float = 100.0
    w_altitude: float = 5.0
    w_position: float = 3.0
    correction_bonus_tier1: float = 1.0
    correction_bonus_tier2: float = 2.0
    correction_bonus_tier3: float = 5.0
    no_correction_penalty: float = -3.0
    cart_threshold: float = 0.8
    stability_penalty: float = -5.0
    altitude_threshold: float = 0.3
    quad_stability_penalty: float = -8.0
    quad_hover_z: float = 1.0
    cart_bonus_thresholds: tuple = (1.0, 2.0, 3.0)
    cart_no_corr_eps: float = 0.1
    quad_bonus_thresholds: tuple = (0.05, 0.1, 0.2)
    quad_no_corr_eps: float = 0.01
    uncert_mag_floor: float = 0.1
    # auto: cartpole if NAME==cartpole; quadrotor_2d if quad 2D; else generic (correction-only).
    # Explicit values avoid mis-parsing state indices for new plants.
    plant_family: str = 'auto'

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> 'AdversarialNominalConfig':
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in dict(data).items() if k in known}
        return cls(**kwargs)


def _theta_limit_cartpole(env) -> float:
    try:
        return float(env.constraints.constraints[0].upper_bounds[2])
    except Exception:
        return 0.2


def _theta_limit_quadrotor(env) -> float:
    try:
        return float(env.constraints.constraints[0].upper_bounds[4])
    except Exception:
        return 0.25


def _resolve_plant_family(env_name: str, env_ref, state_dim: int, cfg: AdversarialNominalConfig) -> str:
    if cfg.plant_family != 'auto':
        return cfg.plant_family
    if env_name == 'cartpole':
        return 'cartpole'
    if env_name == 'quadrotor':
        try:
            from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
            qt = getattr(env_ref, 'QUAD_TYPE', None)
            if qt == QuadType.TWO_D:
                return 'quadrotor_2d'
        except Exception:
            pass
        if state_dim == 6:
            return 'quadrotor_2d'
        return 'generic'
    return 'generic'


def _accumulate_correction_terms(
        r: float,
        uncert_action: Optional[np.ndarray],
        cert_action: Optional[np.ndarray],
        cfg: AdversarialNominalConfig,
        bonus_thresholds: tuple,
        no_corr_eps: float,
) -> float:
    if uncert_action is not None and cert_action is not None:
        uncert_mag = float(np.linalg.norm(np.atleast_1d(uncert_action)))
        correction = float(np.linalg.norm(np.atleast_1d(uncert_action) - np.atleast_1d(cert_action)))
        if cfg.use_correction_reward:
            r += cfg.w_correction * correction
        if cfg.use_correction_ratio and uncert_mag > cfg.uncert_mag_floor:
            r += cfg.w_correction_ratio * (correction / uncert_mag)
        if cfg.use_correction_bonus:
            t1, t2, t3 = bonus_thresholds
            if correction > t1:
                r += cfg.correction_bonus_tier1
            if correction > t2:
                r += cfg.correction_bonus_tier2
            if correction > t3:
                r += cfg.correction_bonus_tier3
        if cfg.use_no_correction_penalty and correction < no_corr_eps:
            r += cfg.no_correction_penalty
    elif uncert_action is not None:
        r += 5.0 * float(np.linalg.norm(np.atleast_1d(uncert_action)))
    return r


def _single_step_raw_reward(
        s_prev: np.ndarray,
        s: np.ndarray,
        info_i: dict,
        uncert_action: Optional[np.ndarray],
        cert_action: Optional[np.ndarray],
        env_name: str,
        env_ref,
        cfg: AdversarialNominalConfig,
) -> float:
    terminated = False
    try:
        ti = info_i.get('terminal_info', {})
        if ti and not ti.get('TimeLimit.truncated', False):
            terminated = True
    except Exception:
        pass
    if terminated:
        return -cfg.w_termination_penalty

    r = 0.0
    family = _resolve_plant_family(env_name, env_ref, int(s.shape[0]), cfg)

    if family == 'generic':
        r = _accumulate_correction_terms(
            r, uncert_action, cert_action, cfg, cfg.quad_bonus_thresholds, cfg.quad_no_corr_eps)
        r += 0.25 * float(np.linalg.norm(s)) + 0.5 * float(np.linalg.norm(s - s_prev))
        return float(r)

    if family == 'quadrotor_2d' and s.shape[0] >= 6:
        x, z, th, thdot = s[0], s[2], s[4], s[5]
        thdot_prev = s_prev[5]
        theta_lim = _theta_limit_quadrotor(env_ref)
        theta_ratio = abs(th) / max(theta_lim, 1e-6)

        r = _accumulate_correction_terms(
            r, uncert_action, cert_action, cfg, cfg.quad_bonus_thresholds, cfg.quad_no_corr_eps)

        if cfg.use_theta_reward:
            r += cfg.w_theta * theta_ratio
        if cfg.use_velocity_reward:
            r += cfg.w_velocity * min(abs(thdot), 2.0)
        if cfg.use_oscillation_reward:
            if np.sign(thdot) != np.sign(thdot_prev) and thdot_prev != 0:
                r += cfg.w_oscillation
        if cfg.use_position_reward:
            position_deviation = np.sqrt(x**2 + (z - cfg.quad_hover_z)**2)
            r += cfg.w_position * min(position_deviation, 2.0)
        if cfg.use_altitude_penalty and z < cfg.altitude_threshold:
            r -= cfg.w_altitude * (cfg.altitude_threshold - z)
        if cfg.use_stability_penalty:
            if theta_ratio < 0.2 and abs(thdot) < 0.3:
                r += cfg.quad_stability_penalty
        return float(r)

    if family == 'cartpole' and s.shape[0] >= 4:
        x, _, th, thdot = s[0], s[1], s[2], s[3]
        thdot_prev = s_prev[3]
        theta_lim = _theta_limit_cartpole(env_ref)
        theta_ratio = abs(th) / max(theta_lim, 1e-6)

        r = _accumulate_correction_terms(
            r, uncert_action, cert_action, cfg, cfg.cart_bonus_thresholds, cfg.cart_no_corr_eps)

        if cfg.use_theta_reward:
            r += cfg.w_theta * theta_ratio
        if cfg.use_velocity_reward:
            r += cfg.w_velocity * min(abs(thdot), 2.0)
        if cfg.use_oscillation_reward:
            if np.sign(thdot) != np.sign(thdot_prev) and thdot_prev != 0:
                r += cfg.w_oscillation
        if cfg.use_cart_penalty:
            if abs(x) > cfg.cart_threshold:
                r -= cfg.w_cart_penalty * (abs(x) - cfg.cart_threshold)
        if cfg.use_stability_penalty:
            if theta_ratio < 0.2 and abs(thdot) < 0.5:
                r += cfg.stability_penalty
        return float(r)

    r = _accumulate_correction_terms(
        r, uncert_action, cert_action, cfg, cfg.cart_bonus_thresholds, cfg.cart_no_corr_eps)
    r += 0.25 * float(np.linalg.norm(s)) + 0.5 * float(np.linalg.norm(s - s_prev))
    return float(r)


def compute_adversarial_rewards_training_batch(
        prev_obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        info: dict,
        uncert_action: Optional[np.ndarray],
        cert_action: Optional[np.ndarray],
        success: bool,
        env_ref,
        cfg: AdversarialNominalConfig,
) -> np.ndarray:
    '''Per-environment adversarial rewards (same batch layout as PPO vector env).'''
    next_obs_b = np.atleast_2d(next_obs)
    prev_obs_b = np.atleast_2d(prev_obs)
    b = next_obs_b.shape[0]
    infos = info.get('n', [info] * b) if isinstance(info, dict) else [dict()] * b
    env_name = getattr(env_ref, 'NAME', 'cartpole')
    out = np.zeros(b, dtype=np.float32)
    for i in range(b):
        ua = uncert_action
        ca = cert_action if success else None
        if ua is not None and getattr(ua, 'ndim', 0) > 1 and ua.shape[0] == b:
            ua = ua[i]
        if ca is not None and getattr(ca, 'ndim', 0) > 1 and ca.shape[0] == b:
            ca = ca[i]
        raw = _single_step_raw_reward(
            prev_obs_b[i], next_obs_b[i], infos[i], ua, ca, env_name, env_ref, cfg)
        tau = max(cfg.temperature, 1e-6)
        out[i] = float(tau * np.log(1.0 + np.exp(raw / tau)))
    return out
