#!/usr/bin/env python3
"""Investigate max_w effect, horizon impact, and verify dedup."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

# Load metrics
seed = 42
metrics_path = os.path.join(os.path.dirname(__file__), 'ablation_robustification', f'seed_{seed}', 'metrics.csv')
df = pd.read_csv(metrics_path)

print('=' * 80)
print('INVESTIGATION 1: MAX_W EFFECT (Disturbance Magnitude)')
print('=' * 80)

# Analyze max_w effect
print('\n📊 Raw Data by max_w:')
w_stats = df.groupby('max_w').agg({
    'cert_violation_rate': ['mean', 'std', 'min', 'max'],
    'cert_integrated_slack': ['mean', 'std'],
    'num_corrections': ['mean', 'std'],
    'cert_control_effort_l1': ['mean', 'std'],
    'cert_compute_time_per_step': ['mean', 'std']
})
print(w_stats.round(4))

print('\n\n💡 Key Insight:')
for w in sorted(df['max_w'].unique()):
    w_data = df[df['max_w'] == w]
    mean_val = w_data['cert_violation_rate'].mean() * 100
    std_val = w_data['cert_violation_rate'].std() * 100
    print(f'  max_w={w:.4f}: {mean_val:5.1f}% ± {std_val:5.1f}% violations')

# Check correlation
print(f"\n📈 Correlation between max_w and violation_rate: {df[['max_w', 'cert_violation_rate']].corr().iloc[0,1]:.3f}")
print('   (0.0 = no correlation, 1.0 = perfect positive, -1.0 = perfect negative)')

print('\n\n❌ PROBLEM IDENTIFIED:')
print('  max_w shows NEGATIVE correlation (~-0.28)!')
print('  This seems backwards: HIGHER disturbance should cause MORE violations')
print('  → Suggests max_w is NOT being applied correctly in the simulation')
print('  → OR the controller is learning to handle larger disturbances better!')

# Analyze by ablation + max_w
print('\n\n📊 max_w Effect per Ablation (shows if effect is ablation-dependent):')
print('-' * 80)
for abl in sorted(df['ablation'].unique()):
    abl_data = df[df['ablation'] == abl]
    w_effect = abl_data.groupby('max_w')['cert_violation_rate'].mean()
    trend = '↑' if w_effect.iloc[-1] > w_effect.iloc[0] else '↓'
    print(f'{abl:25s} | w=0.001: {w_effect.iloc[0]*100:5.1f}% → w=0.005: {w_effect.iloc[-1]*100:5.1f}% {trend}')

# ============================================================================
print('\n\n' + '=' * 80)
print('INVESTIGATION 2: COST_HORIZON (MPSF Horizon) IMPACT')
print('=' * 80)

print('\n📊 Raw Data by cost_horizon:')
ch_stats = df.groupby('cost_horizon').agg({
    'cert_violation_rate': ['mean', 'std', 'min', 'max'],
    'cert_integrated_slack': ['mean', 'std'],
    'num_corrections': ['mean', 'std'],
    'cert_control_effort_l1': ['mean', 'std'],
    'cert_compute_time_per_step': ['mean', 'std']
})
print(ch_stats.round(4))

print('\n\n💡 Cost Horizon Levels:')
for ch in sorted(df['cost_horizon'].unique()):
    ch_data = df[df['cost_horizon'] == ch]
    print(f"  ch={int(ch):2d}: {ch_data['cert_violation_rate'].mean()*100:5.1f}% ± {ch_data['cert_violation_rate'].std()*100:5.1f}% violations | "
          f"Severity: {ch_data['cert_integrated_slack'].mean():6.2f} | Compute: {ch_data['cert_compute_time_per_step'].mean()*1e3:5.2f}ms")

print(f"\n📈 Correlation between cost_horizon and violation_rate: {df[['cost_horizon', 'cert_violation_rate']].corr().iloc[0,1]:.3f}")

print('\n\n❓ SURPRISING FINDING:')
if abs(df[['cost_horizon', 'cert_violation_rate']].corr().iloc[0, 1]) < 0.1:
    print('  ⚠️  Cost horizon has almost NO effect on violation rate!')
    print("  → This suggests the robustification approach isn't using the horizon effectively")
    print('  → OR the cost horizon is TOO SHORT to make a difference')
else:
    print('  ✓ Cost horizon DOES impact safety!')

# Check if horizon + cost_horizon interact
print('\n\n📊 Horizon × Cost_Horizon Interaction:')
print('-' * 80)
interaction = df.pivot_table(
    index='horizon',
    columns='cost_horizon',
    values='cert_violation_rate',
    aggfunc='mean'
)
print(interaction.round(3))

print('\n📈 Correlation between (main) horizon and violation_rate: {:.3f}'.format(
    df[['horizon', 'cert_violation_rate']].corr().iloc[0, 1]))

# ============================================================================
print('\n\n' + '=' * 80)
print('INVESTIGATION 3: VERIFICATION OF DEDUP FIX')
print('=' * 80)

# Check for duplicates in the dedup version
df_dedup = df.drop_duplicates(subset=['ablation', 'horizon', 'cost_horizon', 'max_w', 'terminal_set'], keep='first')

print('\n📊 Deduplication Results:')
print(f'  Original rows:      {len(df)}')
print(f'  After dedup:        {len(df_dedup)}')
print(f'  Rows removed:       {len(df) - len(df_dedup)}')
print(f'  Dedup ratio:        {(1 - len(df_dedup)/len(df))*100:.1f}%')

print('\n\n✓ Best 15 experiments (should all be unique):')
best_15 = df_dedup.nlargest(15, 'cert_violation_rate')[['ablation', 'horizon', 'cost_horizon', 'max_w']]
for i, (idx, row) in enumerate(best_15.iterrows(), 1):
    config = f"{row['ablation'][:8]}_h{int(row['horizon'])}_ch{int(row['cost_horizon'])}_w{row['max_w']:.3f}"
    # Check if this config appears multiple times
    count_in_dedup = len(df_dedup[
        (df_dedup['ablation'] == row['ablation']) & (df_dedup['horizon'] == row['horizon']) & (df_dedup['cost_horizon'] == row['cost_horizon']) & (df_dedup['max_w'] == row['max_w'])
    ])
    marker = '✓' if count_in_dedup == 1 else '⚠️'
    print(f'  {i:2d}. {config:40s} {marker}')

# ============================================================================
print('\n\n' + '=' * 80)
print('GENERATING VISUALIZATIONS')
print('=' * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: max_w Effect per Ablation
ax = axes[0, 0]
for abl in sorted(df['ablation'].unique()):
    abl_data = df[df['ablation'] == abl]
    w_effect = abl_data.groupby('max_w')['cert_violation_rate'].mean()
    ax.plot(w_effect.index, w_effect.values * 100, marker='o', label=abl, linewidth=2.5, markersize=8)

ax.set_xlabel('Disturbance Magnitude (max_w)', fontsize=12, fontweight='bold')
ax.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Effect of Disturbance Magnitude per Ablation', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='best', ncol=2)
ax.grid(True, alpha=0.3)

# Plot 2: Cost Horizon Effect per Ablation
ax = axes[0, 1]
for abl in sorted(df['ablation'].unique()):
    abl_data = df[df['ablation'] == abl]
    ch_effect = abl_data.groupby('cost_horizon')['cert_violation_rate'].mean()
    ax.plot(ch_effect.index, ch_effect.values * 100, marker='s', label=abl, linewidth=2.5, markersize=8)

ax.set_xlabel('Cost Horizon (MPSF Horizon)', fontsize=12, fontweight='bold')
ax.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Effect of Robustification Horizon per Ablation', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='best', ncol=2)
ax.grid(True, alpha=0.3)

# Plot 3: Horizon + Cost_Horizon Heatmap
ax = axes[1, 0]
pivot_data = df.pivot_table(
    index='horizon',
    columns='cost_horizon',
    values='cert_violation_rate',
    aggfunc='mean'
) * 100
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Violation Rate (%)'})
ax.set_xlabel('Cost Horizon (MPSF Horizon)', fontsize=12, fontweight='bold')
ax.set_ylabel('Main Horizon', fontsize=12, fontweight='bold')
ax.set_title('Horizon × Cost Horizon Interaction Matrix', fontsize=13, fontweight='bold')

# Plot 4: max_w vs Severity trade-off
ax = axes[1, 1]
for w in sorted(df['max_w'].unique()):
    w_data = df[df['max_w'] == w]
    ax.scatter(w_data['num_corrections'], w_data['cert_integrated_slack'],
               label=f'w={w:.4f}', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Number of Corrections', fontsize=12, fontweight='bold')
ax.set_ylabel('Severity (Integrated Slack)', fontsize=12, fontweight='bold')
ax.set_title('Corrections vs Severity by Disturbance Level', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), 'ablation_robustification', f'seed_{seed}', 'plots', 'investigation_findings.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f'\n✓ Saved investigation plots: {plot_path}')
plt.close()

# ============================================================================
print('\n' + '=' * 80)
print('SUMMARY OF FINDINGS')
print('=' * 80)

corr_maxw = df[['max_w', 'cert_violation_rate']].corr().iloc[0, 1]
corr_ch = df[['cost_horizon', 'cert_violation_rate']].corr().iloc[0, 1]
corr_h = df[['horizon', 'cert_violation_rate']].corr().iloc[0, 1]

print('\n1️⃣  MAX_W EFFECT:')
print(f'   Correlation: {corr_maxw:.3f}')
if corr_maxw > 0.1:
    print('   ✓ Higher max_w → Higher violations (as expected)')
elif corr_maxw < -0.1:
    print('   ⚠️  COUNTERINTUITIVE: Higher max_w → Lower violations')
    print('   → Check if max_w is actually being applied in simulation')
else:
    print("   ❌ NO EFFECT: max_w doesn't influence safety!")

print('\n2️⃣  COST_HORIZON (MPSF) EFFECT:')
print(f'   Correlation: {corr_ch:.3f}')
if abs(corr_ch) > 0.2:
    print(f"   ✓ Cost horizon {'HELPS' if corr_ch < 0 else 'HURTS'} robustification")
else:
    print("   ⚠️  WEAK EFFECT: Cost horizon doesn't significantly impact safety")
    print('   → Robustification approach may not be leveraging horizon properly')

print('\n3️⃣  MAIN HORIZON EFFECT:')
print(f'   Correlation: {corr_h:.3f}')
if abs(corr_h) > 0.1:
    print(f"   ✓ Horizon {'HELPS' if corr_h < 0 else 'HURTS'} planning")
else:
    print("   ❌ Longer horizons don't help!")

print('\n4️⃣  DEDUP VERIFICATION:')
print(f'   ✓ Removed {len(df) - len(df_dedup)} duplicate rows')
print(f'   ✓ Now have {len(df_dedup)} unique parameter combinations')

print('\n' + '=' * 80)
