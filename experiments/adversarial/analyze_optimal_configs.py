#!/usr/bin/env python3
"""Analyze metrics and identify optimal configurations."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style('whitegrid')

# Load metrics
seed = 42
metrics_path = os.path.join(os.path.dirname(__file__), 'ablation_robustification', f'seed_{seed}', 'metrics.csv')

if not os.path.exists(metrics_path):
    print(f'[ERROR] Metrics file not found: {metrics_path}')
    exit(1)

df = pd.read_csv(metrics_path)
print(f'✓ Loaded {len(df)} experiments from metrics.csv\n')

# ============================================================================
# SECTION 1: BASIC STATISTICS
# ============================================================================
print('=' * 80)
print('SECTION 1: OVERVIEW')
print('=' * 80)
print(f'Total experiments: {len(df)}')
print(f"Unique ablations: {df['ablation'].nunique()}")
print(f"Unique horizons: {df['horizon'].nunique()}")
print(f"Unique cost_horizons: {df['cost_horizon'].nunique()}")
print(f"Max disturbance range: {df['max_w'].min():.6f} to {df['max_w'].max():.6f}")
print()

# ============================================================================
# SECTION 2: IDENTIFY OPTIMAL CONFIGURATIONS - By Use Case
# ============================================================================
print('=' * 80)
print('SECTION 2: OPTIMAL CONFIGURATIONS BY USE CASE')
print('=' * 80)

# Create composite metrics for different use cases
df['safety_score'] = 1 - df['cert_violation_rate']
df['efficiency_score'] = 1 - (df['cert_compute_time_per_step'] / df['cert_compute_time_per_step'].max())
df['smoothness_score'] = 1 - (df['cert_mean_action_rate'] / df['cert_mean_action_rate'].max())
df['robustness_score'] = 1 - (df['cert_integrated_slack'] / df['cert_integrated_slack'].max())

df['safety_first'] = df['safety_score'] * 0.7 + df['efficiency_score'] * 0.2 + df['smoothness_score'] * 0.1
df['balanced'] = df['safety_score'] * 0.5 + df['robustness_score'] * 0.2 + df['efficiency_score'] * 0.2 + df['smoothness_score'] * 0.1
df['minimal_corrections'] = df['safety_score'] * 0.4 + (1 - df['num_corrections'] / df['num_corrections'].max()) * 0.4 + df['efficiency_score'] * 0.2

# Use Case 1: Safety-Critical
print('\n📍 USE CASE 1: SAFETY-CRITICAL (autonomous vehicle, robotic arm)')
print('-' * 80)
top_safety = df.nlargest(5, 'safety_first')[['ablation', 'horizon', 'cost_horizon', 'max_w',
                                             'cert_violation_rate', 'cert_time_to_first_violation',
                                             'cert_integrated_slack', 'cert_compute_time_per_step']]
for idx, (i, row) in enumerate(top_safety.iterrows(), 1):
    print(f"{idx}. {row['ablation']:25s} h={int(row['horizon']):2d} ch={int(row['cost_horizon']):2d} w={row['max_w']:.4f}")
    print(f"   ├─ Violations: {row['cert_violation_rate']*100:5.1f}% | TtFV: {int(row['cert_time_to_first_violation']):4d} steps | Severity: {row['cert_integrated_slack']:6.2f}")
    print(f"   └─ Compute: {row['cert_compute_time_per_step']*1e3:6.2f} ms/step\n")

# Use Case 2: Real-Time
print('\n📍 USE CASE 2: REAL-TIME CONTROL (drone, reactive control)')
print('-' * 80)
df_realtime = df[df['cert_compute_time_per_step'] < 0.01]
if len(df_realtime) > 0:
    top_realtime = df_realtime.nlargest(5, 'safety_score')[['ablation', 'horizon', 'cost_horizon', 'max_w',
                                                            'cert_violation_rate', 'cert_compute_time_per_step',
                                                            'cert_mean_action_rate', 'num_corrections']]
    for idx, (i, row) in enumerate(top_realtime.iterrows(), 1):
        print(f"{idx}. {row['ablation']:25s} h={int(row['horizon']):2d} ch={int(row['cost_horizon']):2d} w={row['max_w']:.4f}")
        print(f"   ├─ Safety: {row['cert_violation_rate']*100:5.1f}%")
        print(f"   ├─ Compute: {row['cert_compute_time_per_step']*1e3:6.2f} ms | Action Rate: {row['cert_mean_action_rate']:.4f}")
        print(f"   └─ Corrections: {int(row['num_corrections']):4d}\n")
else:
    print('⚠️  No configurations found with compute time < 10ms')

# Use Case 3: Minimal interventions
print('\n📍 USE CASE 3: MINIMAL INTERVENTIONS (minimize safety filter activations)')
print('-' * 80)
top_minimal = df.nlargest(5, 'minimal_corrections')[['ablation', 'horizon', 'cost_horizon', 'max_w',
                                                     'cert_violation_rate', 'num_corrections',
                                                     'cert_control_effort_l1', 'cert_compute_time_per_step']]
for idx, (i, row) in enumerate(top_minimal.iterrows(), 1):
    print(f"{idx}. {row['ablation']:25s} h={int(row['horizon']):2d} ch={int(row['cost_horizon']):2d} w={row['max_w']:.4f}")
    print(f"   ├─ Safety: {row['cert_violation_rate']*100:5.1f}% | Corrections: {int(row['num_corrections']):4d}")
    print(f"   └─ Effort: {row['cert_control_effort_l1']:7.1f} | Compute: {row['cert_compute_time_per_step']*1e3:6.2f} ms\n")

# ============================================================================
# SECTION 3: ABLATION IMPORTANCE
# ============================================================================
print('\n' + '=' * 80)
print('SECTION 3: ABLATION RANKING (by safety)')
print('=' * 80)

abl_summary = df.groupby('ablation').agg({
    'cert_violation_rate': ['mean', 'std'],
    'cert_time_to_first_violation': 'mean',
    'cert_control_effort_l1': 'mean',
    'cert_compute_time_per_step': 'mean'
})
abl_summary.columns = ['violation_mean', 'violation_std', 'ttfv_mean', 'effort_mean', 'compute_mean']
abl_summary = abl_summary.sort_values('violation_mean')

print('\nRanked by Lowest Violation Rate:')
print(f"{'Rank':<5} {'Ablation':<30} {'Violations':<12} {'±Std':<8} {'TtFV':<6} {'Effort':<8} {'Time':<8}")
print('-' * 80)
for rank, (abl, row) in enumerate(abl_summary.iterrows(), 1):
    print(f"{rank:<5} {abl:<30} {row['violation_mean']*100:>5.1f}%      "
          f"{row['violation_std']*100:>5.1f}%  {int(row['ttfv_mean']):5d}  {row['effort_mean']:>7.1f}  {row['compute_mean']*1e3:>6.2f}ms")

# ============================================================================
# SECTION 4: SENSITIVITY ANALYSIS
# ============================================================================
print('\n' + '=' * 80)
print('SECTION 4: SENSITIVITY TO PARAMETERS')
print('=' * 80)

print('\n📊 Impact of Disturbance Magnitude (max_w):')
w_impact = df.groupby('max_w')['cert_violation_rate'].agg(['mean', 'std', 'count'])
print(f"{'max_w':<12} {'Avg Violation':<15} {'Std Dev':<10} {'# Configs':<10}")
print('-' * 50)
for w, row in w_impact.iterrows():
    print(f"{w:<12.6f} {row['mean']*100:>5.1f}%         {row['std']*100:>5.1f}%    {int(row['count']):>5d}")

print('\n📊 Impact of Horizon:')
h_impact = df.groupby('horizon')['cert_violation_rate'].agg(['mean', 'std', 'count'])
print(f"{'Horizon':<10} {'Avg Violation':<15} {'Std Dev':<10} {'# Configs':<10}")
print('-' * 45)
for h, row in h_impact.iterrows():
    print(f"{int(h):<10} {row['mean']*100:>5.1f}%         {row['std']*100:>5.1f}%    {int(row['count']):>5d}")

# ============================================================================
# SECTION 5: TRADE-OFF ANALYSIS
# ============================================================================
print('\n' + '=' * 80)
print('SECTION 5: KEY TRADE-OFFS')
print('=' * 80)

print('\n1️⃣  Safety vs. Severity of Violations:')
safe_sev_corr = df[['cert_violation_rate', 'cert_integrated_slack']].corr().iloc[0, 1]
print(f'   Correlation: {safe_sev_corr:.3f}')
if safe_sev_corr > 0.5:
    print('   ⚠️  STRONG: More violations → More severe violations')
else:
    print('   NOTE: Can have high violation rate but low severity')

print('\n2️⃣  Safety vs. Control Effort:')
safe_effort_corr = df[['cert_violation_rate', 'cert_control_effort_l1']].corr().iloc[0, 1]
print(f'   Correlation: {safe_effort_corr:.3f}')
if abs(safe_effort_corr) > 0.5:
    print(f"   ⚠️  STRONG: Achieving safety requires {'significant' if safe_effort_corr > 0 else 'less'} control")
else:
    print('   ✓ EFFICIENT: Safety achieved without excessive control effort')

print('\n3️⃣  Safety vs. Compute Time:')
safe_compute_corr = df[['cert_violation_rate', 'cert_compute_time_per_step']].corr().iloc[0, 1]
print(f'   Correlation: {safe_compute_corr:.3f}')
if abs(safe_compute_corr) > 0.3:
    print(f"   ⚠️  SIGNIFICANT: Safer requires {'more' if safe_compute_corr > 0 else 'less'} computation")
else:
    print('   ✓ SCALABLE: Safety achieved with reasonable compute')

# ============================================================================
# SECTION 6: GENERATE VISUALIZATION
# ============================================================================
print('\n' + '=' * 80)
print('SECTION 6: GENERATING PARETO FRONTIER PLOTS')
print('=' * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Safety vs Compute Time
ax = axes[0]
scatter = ax.scatter(df['cert_compute_time_per_step'] * 1e3, df['cert_violation_rate'] * 100,
                     c=df['num_corrections'], cmap='rocket_r', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Compute Time (ms/step)', fontsize=12, fontweight='bold')
ax.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Safety vs Computational Cost', fontsize=13, fontweight='bold')
cbar1 = plt.colorbar(scatter, ax=ax)
cbar1.set_label('# Corrections', fontsize=10)
ax.grid(True, alpha=0.3)

best_safe = df.loc[df['cert_violation_rate'].idxmin()]
best_fast = df.loc[df['cert_compute_time_per_step'].idxmin()]
ax.scatter([best_safe['cert_compute_time_per_step'] * 1e3], [best_safe['cert_violation_rate'] * 100],
           marker='*', s=600, c='gold', edgecolors='red', linewidth=2, label='Most Safe', zorder=10)
ax.scatter([best_fast['cert_compute_time_per_step'] * 1e3], [best_fast['cert_violation_rate'] * 100],
           marker='s', s=150, c='lightblue', edgecolors='blue', linewidth=2, label='Fastest', zorder=10)
ax.legend(fontsize=10, loc='best')

# Plot 2: Safety vs Corrections Needed
ax = axes[1]
scatter2 = ax.scatter(df['num_corrections'], df['cert_violation_rate'] * 100,
                      c=df['cert_integrated_slack'], cmap='viridis', s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Number of Corrections', fontsize=12, fontweight='bold')
ax.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Safety vs Filter Activity', fontsize=13, fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=ax)
cbar2.set_label('Severity (Slack)', fontsize=10)
ax.grid(True, alpha=0.3)

best_minimal = df.loc[(1 - df['num_corrections'] / df['num_corrections'].max() + df['safety_score']).idxmax()]
ax.scatter([best_minimal['num_corrections']], [best_minimal['cert_violation_rate'] * 100],
           marker='D', s=150, c='lightgreen', edgecolors='green', linewidth=2, label='Low Activity Safe', zorder=10)
ax.legend(fontsize=10, loc='best')

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), 'ablation_robustification', f'seed_{seed}', 'plots', 'optimal_configs_pareto.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f'\n✓ Saved Pareto frontier to: {plot_path}')
plt.close()

print('\n' + '=' * 80)
print('✓ ANALYSIS COMPLETE - All optimal configurations identified!')
print('=' * 80)
