# Robustification Metrics Documentation

This document explains all metrics computed and visualized in the robustification ablation studies.

---

## Safety Metrics

### 1. **Violation Rate** (Certified & Uncertified)

**Metric Names:**
- `cert_violation_rate` (Certified)
- `uncert_violation_rate` (Uncertified)

**Definition:**
Fraction of timesteps where the system state violates safety constraints. For CartPole, this is when the angle θ exceeds bounds: θ > θ_ub or θ < θ_lb.

**Formula:**
```
violation_rate = (# timesteps with violation) / (total # timesteps)
```

**Interpretation:**
- **0.0** = Perfect safety (no violations)
- **1.0** = Unsafe (violated at every timestep)
- Certified = with the safety filter applied
- Uncertified = without the safety filter (baseline control alone)

**Why it matters:**
Core safety metric. Higher violation rates indicate less safe control strategies.

---

### 2. **Episode Violation Rate** (Certified & Uncertified)

**Metric Names:**
- `cert_episode_violation_rate`
- `uncert_episode_violation_rate`

**Definition:**
Fraction of episodes where at least one violation occurred.

**Formula:**
```
episode_violation_rate = (# episodes with ≥1 violation) / (total # episodes)
```

**Interpretation:**
- **0.0** = All episodes remain safe (no episode violated)
- **1.0** = Every episode had a violation
- Useful for understanding how often violations are unavoidable

**Why it matters:**
Distinguishes between isolated violations vs. systematic unsafety. An episode violation rate of 0% is ideal but rare in adversarial settings.

---

### 3. **Time to First Violation (TtFV)**

**Metric Names:**
- `cert_time_to_first_violation` (Certified)
- `uncert_time_to_first_violation` (Uncertified)

**Definition:**
Number of timesteps until the first constraint violation occurs (minimum across all episodes). If no violation occurs, the value is the episode horizon.

**Formula:**
```
ttfv = min(timestep where violation first occurs, across all episodes)
```

**Interpretation:**
- Higher values = System remains safe for longer
- Maximum possible = episode horizon (e.g., 500 steps)
- Lower values = System violates constraints quickly
- **Current issue:** Uses `min()` across episodes, so shows worst-case (earliest violation)

**Why it matters:**
Indicates robustness: how long the system can operate safely before inevitable failure in adversarial scenarios.

---

### 4. **Integrated Slack (Severity)**

**Metric Names:**
- `cert_integrated_slack` (Certified)
- `uncert_integrated_slack` (Uncertified)

**Definition:**
Sum of all constraint violation magnitudes across all timesteps. Measures total severity of violations, not just frequency.

**Formula:**
```
severity = Σ max(0, θ - θ_ub) + max(0, θ_lb - θ)  [over all timesteps]
```

**Interpretation:**
- **0** = No violations (safe)
- Higher values = More severe violations (state went far outside bounds)
- A low violation rate with high severity = few violations, but they're severe

**Why it matters:**
Distinguishes between minor constraint breaches and dangerous violations. Two strategies with the same violation rate might have very different severity levels.

---

### 5. **Max Violation**

**Metric Names:**
- `cert_max_violation`
- `uncert_max_violation`

**Definition:**
The maximum constraint violation magnitude across all timesteps and episodes.

**Formula:**
```
max_violation = max(θ - θ_ub, θ_lb - θ)  [over all timesteps]
```

**Interpretation:**
- **0** = Safe
- Higher = Worse single worst-case violation
- Worst-case analysis metric

**Why it matters:**
Safety-critical systems must bound worst-case behavior. Shows if any individual violation is dangerously large.

---

## Control Effort & Smoothness Metrics

### 6. **Control Effort (L1 & L2 Norm)**

**Metric Names:**
- `cert_control_effort_l1`
- `uncert_control_effort_l1`
- `cert_control_effort_l2`
- `uncert_control_effort_l2`

**Definition:**
Total magnitude of control inputs applied over the episode.

**Formula:**
```
L1 effort = Σ |u_t|  [over all timesteps]
L2 effort = √(Σ u_t²) [over all timesteps]
```

**Interpretation:**
- Higher = More aggressive control (more energy used)
- Lower = More conservative, smoother control
- L1 considers individual action magnitudes; L2 is Euclidean norm

**Why it matters:**
Practical concern: high control effort causes:
- Motor wear and tear
- Energy consumption
- Actuator saturation
- Reduced lifespan of hardware

Trade-off: Safety filters need to act, requiring control effort.

---

### 7. **Action Rate (Δu)**

**Metric Names:**
- `cert_mean_action_rate`
- `uncert_mean_action_rate`
- `cert_max_action_rate`
- `uncert_max_action_rate`

**Definition:**
Rate of change of control actions: acceleration/deceleration of the control signal.

**Formula:**
```
Δu_t = |u_t - u_{t-1}|  (timestep-wise change)
mean_action_rate = (Σ Δu_t) / T
max_action_rate = max(Δu_t)
```

**Interpretation:**
- Lower = Smoother, less jerky control
- Higher = Abrupt, reactive corrections
- Important for mechanical systems where rapid changes cause stress

**Why it matters:**
Smoothness metric. Safety filters can cause sudden corrections that:
- Create mechanical stress
- Feel uncomfortable (robot)
- Trigger alarms in sensitive equipment

---

## Filter Activity & Correction Metrics

### 8. **Number of Corrections**

**Metric Name:**
- `num_corrections`

**Definition:**
Total count of timesteps where the safety filter made a non-trivial correction (|correction| > 1e-6).

**Formula:**
```
num_corrections = count(timesteps where |correction| > threshold)
```

**Interpretation:**
- **0** = Filter never intervened (control was already safe)
- Higher = Filter had to correct frequently
- If high, suggests baseline control is unsafe

**Why it matters:**
Indicates workload on the safety filter. High correction counts mean:
- The baseline controller is often unsafe
- The filter must be active/responsive
- Potential bottleneck for real-time systems

---

### 9. **Correction Magnitude Statistics**

**Metric Names:**
- `total_correction_magnitude` (sum of all corrections)
- `max_correction` (largest single correction)
- `mean_correction` (average correction size)

**Definition:**
Aggregate statistics of correction vector magnitudes.

**Formula:**
```
correction_magnitude_t = ||correction_t||  [Euclidean norm]
```

**Interpretation:**
- **Max Correction**: Largest needed intervention (worst-case)
- **Mean Correction**: Typical intervention size
- **Total Magnitude**: Cumulative correction effort

**Why it matters:**
Distinguishes filter types:
- Large corrections, few times = reactive filter (only acts when needed)
- Small corrections, frequent = proactive filter (continuous guiding)

---

### 10. **Correction Rate**

**Metric Name:**
- `correction_rate`

**Definition:**
Fraction of timesteps where a correction occurred.

**Formula:**
```
correction_rate = (# timesteps with correction) / (total timesteps)
```

**Interpretation:**
- **0.0** = Never corrected
- **1.0** = Corrected at every step
- 0.0-0.1 = Rarely needed (baseline mostly safe)
- 0.3+ = Frequently intervening (baseline often unsafe)

**Why it matters:**
Indicates filter necessity. A low correction rate suggests:
- Good baseline control
- Or the control is already constrained

A high rate suggests:
- Baseline needs help
- Or the safety specification is very tight

---

## Compute Metrics

### 11. **Compute Time**

**Metric Names:**
- `cert_compute_time_total` (total across all timesteps)
- `cert_compute_time_per_step` (per-step average)
- `uncert_compute_time_total`
- `uncert_compute_time_per_step`

**Definition:**
Wall-clock time to compute control decision (filter + MPC).

**Interpretation:**
- **Certified** = With safety filter active
- **Uncertified** = Without filter (baseline MPC only)
- Per-step averaged over episode horizon

**Typical ranges:**
- < 1 ms = Real-time capable (fast systems)
- 1-10 ms = Most industrial applications
- > 100 ms = Non-real-time (offline planning)

**Why it matters:**
Real-time feasibility constraint. A safe control that's too slow is useless in dynamic environments.

---

## Performance Metrics

### 12. **RMSE (Root Mean Square Error)**

**Metric Names:**
- `rmse_cert` (Certified)
- `rmse_uncert` (Uncertified)

**Definition:**
Tracking error: how well the system follows desired trajectory/setpoint.

**Formula:**
```
RMSE = √( (1/T) Σ ||x_t - x_target||² )
```

**Interpretation:**
- **0** = Perfect tracking
- Higher = Worse tracking performance
- Core performance metric

**Why it matters:**
Fundamental trade-off: safety vs. performance.
- Aggressive controllers track better but may violate constraints
- Conservative/safe controllers may undershoot performance

---

## Summary of Trade-offs

| Metric | Better Value | Trade-off Notes |
|--------|--------------|-----------------|
| Violation Rate | Lower (0%) | vs. Control Effort |
| TtFV | Higher | vs. Responsive corrections |
| Severity | Lower | vs. Constraint tightness |
| Control Effort | Lower | vs. Safety margins |
| Action Rate | Lower | vs. Reactiveness |
| Corrections | Fewer | vs. Constraint tightness |
| Compute Time | Lower | vs. Solution quality |
| RMSE | Lower | vs. Safety margins |

---

## Metric Categories

### Primary Safety Metrics
*Focus on constraint satisfaction:*
- Violation Rate
- Episode Violation Rate
- Time to First Violation
- Integrated Slack
- Max Violation

### Secondary Safety Indicators
*Indicate filter activity/workload:*
- Number of Corrections
- Correction Magnitude
- Correction Rate

### Performance/Cost Metrics
*Measure computational and control costs:*
- Control Effort (L1, L2)
- Action Rate
- Compute Time
- RMSE

---

## How to Interpret the Visualization Plots

### **Robustification Metrics Summary (8 panels)**
- Top-left to mid-right: Primary safety and magnitude metrics
- Bottom: Computational efficiency and performance trade-offs

### **Best/Worst Performers**
- Green bars (best) = Most adversarial, highest violations but potentially robust
- Red bars (worst) = Least adversarial, safest but may be conservative

### **Interaction Effects (4 panels)**
- Shows how each ablation responds to varying:
  - Disturbance magnitude
  - Planning horizon
  - Filter workload

### **Heatmaps (6 panels)**
- Ablation vs. parameters colored by metric value
- Reveals which combinations are optimal for each metric

---

## Recommendations for Analysis

1. **Safety First**: Focus on violation-related metrics (rate, severity, TtFV)
2. **Feasibility Check**: Ensure compute time is acceptable
3. **Find Sweet Spot**: Balance safety vs. performance/effort
4. **Robustness**: Prefer strategies with stable metrics across parameter ranges
5. **Worst-Case**: Don't ignore max violation or maximum compute time

---

## Metric Computation Details

All metrics are computed in:
- **Python File**: `robustification_metrics.py`
- **Function**: `compute_robustification_metrics()`
- **Output**: Dictionary with all metrics + CSV export

Metrics are aggregated:
- **Per-experiment**: One row per (ablation, horizon, max_w, cost_horizon, terminal_set) combination
- **Stored in**: `metrics.csv` and related files
- **Updated by**: `eval_ablation_robustification.py` and plotting scripts
