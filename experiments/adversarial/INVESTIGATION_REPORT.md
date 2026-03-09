# Investigation Report: Parameter Effects in Robustification

**Date:** March 5, 2026
**Analysis of:** 216 experiments across 8 ablations, 3 horizons, 3 cost_horizons, 3 disturbance levels

---

## Executive Summary

### 🚨 CRITICAL FINDINGS

1. **max_w (Disturbance) has WEAK effect** (r=0.125)
   - Expected: Higher disturbance → More violations
   - Actual: max_w=0.002 gives safest results (38.4%), max_w=0.005 gives worst (44.8%)
   - **Implication:** Disturbance magnitude is barely being applied or model is learning to handle larger disturbances

2. **cost_horizon (MPSF Horizon) has ZERO effect** (r=0.000)
   - ALL cost_horizon values (6, 8, 10) produce IDENTICAL results: 41.46% violation rate
   - **Critical Issue:** Robustification approach is NOT WORKING
   - **Implication:** Parameter is being ignored or horizon is too short to matter

3. **Main horizon has negligible effect** (r=-0.048)
   - Longer planning doesn't improve safety
   - h=40 barely better than h=20
   - **Implication:** Planning horizon alone doesn't solve the safety problem

4. **Three models appeared repeated in plots** (NOT actually repeated)
   - Each parameter combination (h, ch, max_w) is a unique experiment
   - Same ablation appears multiple times because it performs well across parameters
   - **Why it looks like repetition:** temp_30, baseline, and no_velocity all achieve high violations consistently

---

## Detailed Analysis

### 1. max_w (Disturbance Magnitude) Investigation

**Data:**
```
max_w=0.001  →  41.2% ± 13.2% violations
max_w=0.002  →  38.4% ± 17.4% violations  ← BEST (unexpected!)
max_w=0.005  →  44.8% ± 17.3% violations  ← WORST
```

**Per-Ablation Response:**
- **Positive responders** (violations increase with max_w): no_correction_mag, no_velocity, state_only, temp_30, w_correction_10
- **Negative responders** (violations decrease with max_w): baseline, no_correction_bonus, no_stability_penalty

**Interpretation:**
The mixed response suggests:
1. Some ablations legitimately struggle more with higher disturbance ✓
2. Some ablations benefit from larger disturbance (unexpected) - possibly because:
   - The disturbance distribution shifts optimization landscape
   - Or disturbance isn't actually being applied uniformly

**Recommendation:** Investigate how `max_w` is injected into the simulation. Is it:
- Applied during training only?
- Applied during evaluation?
- Applied to all states or only specific ones?

---

### 2. cost_horizon (MPSF Robustification Horizon) Investigation

**Data:**
```
cost_horizon = 6  →  41.46% ± 16.34% violations
cost_horizon = 8  →  41.46% ± 16.34% violations  ← IDENTICAL
cost_horizon = 10 →  41.46% ± 16.34% violations  ← IDENTICAL
```

**Per-Ablation Response:**
All ablations show FLAT lines across cost_horizon values. No variance.

**Interpretation:**
This is the smoking gun 🎯
- The robustification horizon (`cost_horizon`) has **ZERO impact** on safety
- Either:
  1. Parameter is not being used in the robustification optimization
  2. Horizon window is so short that it makes no difference
  3. Robustification algorithm has a bug
  4. Cost horizon already saturated at minimum value

**Critical Action Item:**
Check `eval_ablation_robustification.py` and `robustification_metrics.py`:
- Is `cost_horizon` passed to the MPC/robustification solver?
- Is the horizon actually being used in constraint/cost function?
- Should it be longer? (Try 20, 30, 50)

---

### 3. Main Horizon Investigation

**Data:**
```
horizon = 20  →  42.6% ± 16.4% violations
horizon = 30  →  41.1% ± 17.5% violations  ← Slight improvement
horizon = 40  →  40.7% ± 15.0% violations  ← Still minimal
```

**Correlation:** -0.048 (essentially zero)

**Per-Ablation Response:**
Some ablations improve with longer horizon, others don't. No consistent pattern.

**Interpretation:**
Longer planning windows provide minimal safety benefit—planning isn't the bottleneck. The issue is likely:
- Constraint tightness (constraints are too tight to satisfy)
- Disturbance magnitude (too large to predict/overcome)
- Controller capability (baseline controller fundamentally unsafe)

---

### 4. "Three Models Repeated" Explanation

**What you observed:**
In the "Best 15" plot, you saw:
- temp_30 appearing 6 times
- baseline appearing 6 times
- no_velocity appearing 3 times

**Why?**
These aren't duplicates—they're the SAME ABLATION evaluated at different parameter combinations:

**temp_30 appears 6 times with configs:**
1. h=40, ch=10, w=0.005 ← Most adversarial configuration found
2. h=40, ch=6, w=0.005
3. h=40, ch=8, w=0.005
4. h=30, ch=10, w=0.005
5. h=30, ch=6, w=0.005
6. h=30, ch=8, w=0.005

**Why it looks repetitive:**
- Parameters (h, ch, w) don't significantly change the outcome
- temp_30, baseline, and no_velocity happen to be the worst ablations
- So they dominate the top-15 list regardless of parameter setting

**This is actually revealing:** The ablation choice matters much more than the parameter tuning.

---

## Heatmap Analysis

**Horizon × Cost_Horizon Interaction Matrix:**
```
                cost_horizon
                 6     8    10
horizon = 20   42.6% 42.6% 42.6%
horizon = 30   41.1% 41.1% 41.1%
horizon = 40   40.7% 40.7% 40.7%
```

**Key observation:** Each ROW is identical (same violation rate across all cost_horizons)

This definitively proves cost_horizon has no effect.

---

## Root Cause Analysis

### Why Is Robustification Not Working?

Three possibilities:

**Hypothesis 1: Cost Horizon Too Short**
- Current range: 6-10 steps
- MPC horizon: 20-40 steps
- The robustification window (6-10) might be too small to make a difference
- Solution: Increase cost_horizon to 20, 30, 50

**Hypothesis 2: Parameter Not Connected**
- cost_horizon might not be passed to the solver
- It could be hard-coded or ignored in `robustification_metrics.py`
- Solution: Check variable flow from input → metrics computation

**Hypothesis 3: Algorithm Limit**
- The robustification approach (worst-case optimization) might have fundamentally limited effectiveness
- Cannot improve beyond ~30-40% violations with this disturbance size
- Solution: Try different robustification strategies (distributionally robust, chance constraints, etc.)

---

## Recommendations & Next Steps

### Immediate Actions (do first):

1. **Verify cost_horizon is actually used**
   ```bash
   # In robustification_metrics.py:
   - Check if cost_horizon parameter is passed to the solver
   - Verify it affects the constraint horizon in MPC
   - Grep for "cost_horizon" usage
   ```

2. **Test with larger cost_horizon values**
   - Try: cost_horizon = 20, 30, 50
   - See if effect emerges with longer robustification window
   - Current 6-10 might be too short

3. **Verify max_w application**
   - Check if disturbance injection is working correctly
   - Confirm it affects all states uniformly
   - Verify magnitude matches expectations

### Secondary Actions (if above doesn't help):

4. **Increase main horizon**
   - Try h=50, 60, 100
   - Current h=20-40 might be insufficient for longer horizons

5. **Change robustification approach**
   - Current: worst-case (min-max)
   - Alternative: distributionally robust optimization
   - Alternative: probabilistic chance constraints (10%, 5% violation rate)

6. **Relax constraints**
   - Current constraints might be impossible to satisfy
   - Test: Wider constraint bounds (±0.3 instead of ±0.2)

---

## Comparison with Expected Behavior

| Parameter | Expected | Actual | Status |
|-----------|----------|--------|--------|
| max_w ↑ causes violations ↑ | r = 0.7 to 1.0 | r = 0.125 | ❌ WEAK |
| cost_horizon ↑ reduces violations | r = -0.5 to -1.0 | r = 0.000 | ❌ BROKEN |
| horizon ↑ improves safety | r = -0.3 to -0.5 | r = -0.048 | ❌ NEGLIGIBLE |
| Parameter combinations vary output | Expected variance | All identical | ❌ INVARIANT |

---

## Visualization Output

Generated 3 detailed investigation plots:

1. **investigation_parameters_effect.png**
   - Shows all three parameter effects with correlations
   - Identifies cost_horizon as zero-effect parameter
   - Boxplots showing data distribution

2. **investigation_ablation_responses.png**
   - Each ablation's response to parameter combinations
   - Shows which ablations respond to disturbance vs which don't

3. **investigation_repetition_explained.png**
   - Explains why "three models repeated" in top-15
   - Shows it's parameter combinations, not duplicate experiments
   - Annotated with clear explanation

---

## Conclusion

The robustification system is **partially broken**:

- ✓ Ablation selection matters (w_correction_10 vs temp_30 differ by 19 percentage points)
- ✓ Disturbance magnitude has weak but real effect
- ❌ **Robustification horizon (cost_horizon) has ZERO effect** — likely unused parameter
- ❌ Main planning horizon barely helps — insufficient for the safety challenge
- ❌ Parameter combinations are largely invariant — suggests overfitting to specific conditions

**Most critical issue:** cost_horizon parameter appears disconnected from the optimization. This should be investigated as priority #1.

---

*Full analysis code: `investigate_findings.py`, `generate_investigation_plots.py`*
