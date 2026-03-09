#!/usr/bin/env python3
"""Create detailed investigation findings report with all visualizations."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

seed = 42
metrics_path = os.path.join(os.path.dirname(__file__), 'ablation_robustification', f'seed_{seed}', 'metrics.csv')
df = pd.read_csv(metrics_path)
plot_dir = os.path.join(os.path.dirname(__file__), 'ablation_robustification', f'seed_{seed}', 'plots')
os.makedirs(plot_dir, exist_ok=True)

# ============================================================================
# COMPREHENSIVE INVESTIGATION PLOTS
# ============================================================================

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Row 1: max_w Effects
ax1 = fig.add_subplot(gs[0, :2])
for abl in sorted(df['ablation'].unique()):
    abl_data = df[df['ablation'] == abl]
    w_effect = abl_data.groupby('max_w')['cert_violation_rate'].mean()
    ax1.plot(w_effect.index * 1000, w_effect.values * 100, marker='o', label=abl, linewidth=2.5, markersize=9)
ax1.set_xlabel('Disturbance Magnitude (max_w × 1000)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('❌ WEAK MAX_W EFFECT: Should increase violations with higher disturbance', fontsize=13, fontweight='bold', color='darkred')
ax1.legend(fontsize=9, loc='best', ncol=4)
ax1.grid(True, alpha=0.4)

# Correlation text
cor_maxw = df[['max_w', 'cert_violation_rate']].corr().iloc[0, 1]
ax1.text(0.5, 0.95, f'Correlation: {cor_maxw:.3f} (WEAK!)', transform=ax1.transAxes,
         fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
         verticalalignment='top', horizontalalignment='center', fontweight='bold')

# Row 1, Col 3: max_w statistics boxplot
ax2 = fig.add_subplot(gs[0, 2])
data_for_box = [df[df['max_w'] == w]['cert_violation_rate'].values * 100 for w in sorted(df['max_w'].unique())]
bp = ax2.boxplot(data_for_box, labels=[f'{w*1000:.1f}' for w in sorted(df['max_w'].unique())], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#ff9999', '#ffcc99', '#99ccff']):
    patch.set_facecolor(color)
ax2.set_xlabel('max_w (× 1000)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Violation Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title('Distribution by max_w', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Row 2: cost_horizon (MPSF) Effects
ax3 = fig.add_subplot(gs[1, :2])
for abl in sorted(df['ablation'].unique()):
    abl_data = df[df['ablation'] == abl]
    ch_effect = abl_data.groupby('cost_horizon')['cert_violation_rate'].mean()
    ax3.plot(ch_effect.index, ch_effect.values * 100, marker='s', label=abl, linewidth=2.5, markersize=9)
ax3.set_xlabel('Cost Horizon (MPSF Horizon)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
ax3.set_title('🚨 CRITICAL: Cost_Horizon has ZERO effect! Robustification not working!', fontsize=13, fontweight='bold', color='darkred')
ax3.legend(fontsize=9, loc='best', ncol=4)
ax3.grid(True, alpha=0.4)

cor_ch = df[['cost_horizon', 'cert_violation_rate']].corr().iloc[0, 1]
ax3.text(0.5, 0.95, f'Correlation: {cor_ch:.3f} (NON-EXISTENT!)', transform=ax3.transAxes,
         fontsize=11, bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
         verticalalignment='top', horizontalalignment='center', fontweight='bold', color='white')

# Row 2, Col 3: cost_horizon statistics
ax4 = fig.add_subplot(gs[1, 2])
ax4.bar(range(len(df['cost_horizon'].unique())),
        df.groupby('cost_horizon')['cert_violation_rate'].mean().values * 100,
        color=['#ff6666' for _ in df['cost_horizon'].unique()])
ax4.set_xticks(range(len(df['cost_horizon'].unique())))
ax4.set_xticklabels(sorted(df['cost_horizon'].unique()))
ax4.set_ylabel('Mean Violation Rate (%)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Cost Horizon', fontsize=11, fontweight='bold')
ax4.set_title('Identical across all cost_horizons!', fontsize=12, fontweight='bold', color='darkred')
ax4.set_ylim([40, 42.5])
ax4.grid(True, alpha=0.3, axis='y')

# Row 3: Main Horizon Effects
ax5 = fig.add_subplot(gs[2, :2])
for abl in sorted(df['ablation'].unique()):
    abl_data = df[df['ablation'] == abl]
    h_effect = abl_data.groupby('horizon')['cert_violation_rate'].mean()
    ax5.plot(h_effect.index, h_effect.values * 100, marker='^', label=abl, linewidth=2.5, markersize=9)
ax5.set_xlabel('Main Planning Horizon', fontsize=12, fontweight='bold')
ax5.set_ylabel('Violation Rate (%)', fontsize=12, fontweight='bold')
ax5.set_title('⚠️  WEAK HORIZON EFFECT: Longer planning barely helps', fontsize=13, fontweight='bold', color='darkred')
ax5.legend(fontsize=9, loc='best', ncol=4)
ax5.grid(True, alpha=0.4)

cor_h = df[['horizon', 'cert_violation_rate']].corr().iloc[0, 1]
ax5.text(0.5, 0.95, f'Correlation: {cor_h:.3f} (near zero)', transform=ax5.transAxes,
         fontsize=11, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7),
         verticalalignment='top', horizontalalignment='center', fontweight='bold')

# Row 3, Col 3: Interaction heatmap
ax6 = fig.add_subplot(gs[2, 2])
pivot_data = df.pivot_table(
    index='horizon',
    columns='cost_horizon',
    values='cert_violation_rate',
    aggfunc='mean'
) * 100
sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax6,
            cbar_kws={'label': 'Violation %'}, vmin=39, vmax=44)
ax6.set_title('No Interaction Pattern', fontsize=12, fontweight='bold')
ax6.set_xlabel('Cost Horizon', fontsize=10, fontweight='bold')
ax6.set_ylabel('Main Horizon', fontsize=10, fontweight='bold')

plt.suptitle('Investigation: Why Robustification Parameters Don\'t Work',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(os.path.join(plot_dir, 'investigation_parameters_effect.png'), dpi=300, bbox_inches='tight')
print('✓ Saved: investigation_parameters_effect.png')
plt.close()

# ============================================================================
# SECOND FIGURE: Ablation-Specific Responses to Parameters
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, abl in enumerate(sorted(df['ablation'].unique())):
    ax = axes[idx]
    abl_data = df[df['ablation'] == abl]

    # Create nested plot showing max_w × cost_horizon
    for w in sorted(df['max_w'].unique()):
        w_data = abl_data[abl_data['max_w'] == w]
        ch_effect = w_data.groupby('cost_horizon')['cert_violation_rate'].mean()
        ax.plot(ch_effect.index, ch_effect.values * 100, marker='o',
                label=f'w={w:.4f}', linewidth=2, markersize=7)

    ax.set_xlabel('Cost Horizon', fontsize=10, fontweight='bold')
    ax.set_ylabel('Violation Rate (%)', fontsize=10, fontweight='bold')
    ax.set_title(f'{abl}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 80])

plt.suptitle('Each Ablation\'s Response to Parameters (cost_horizon × max_w)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'investigation_ablation_responses.png'), dpi=300, bbox_inches='tight')
print('✓ Saved: investigation_ablation_responses.png')
plt.close()

# ============================================================================
# THIRD FIGURE: Why Three Experiments Looked Repeated
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 15 by violation rate
best_exp = df.nlargest(15, 'cert_violation_rate')[['ablation', 'horizon', 'cost_horizon', 'max_w', 'cert_violation_rate']]
ax = axes[0]
colors = plt.cm.Set3(np.arange(best_exp['ablation'].nunique()))
color_map = {abl: colors[i] for i, abl in enumerate(sorted(best_exp['ablation'].unique()))}

bar_colors = [color_map[abl] for abl in best_exp['ablation']]
ax.barh(range(len(best_exp)), best_exp['cert_violation_rate'].values * 100, color=bar_colors, edgecolor='black', linewidth=1)
ax.set_yticks(range(len(best_exp)))
labels = [f"{row['ablation'][:8]}_h{int(row['horizon'])}_ch{int(row['cost_horizon'])}_w{row['max_w']:.3f}"
          for _, row in best_exp.iterrows()]
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Violation Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Best 15 Experiments: Why "Three Models Repeated"?', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Annotation explaining the repetition
ablation_counts = best_exp['ablation'].value_counts()
text_str = 'Why same ablations appear multiple times:\n'
for abl, count in ablation_counts.items():
    text_str += f'• {abl}: {count} different parameter combos (h, ch, w vary)\n'
text_str += '\nThis is EXPECTED - each parameter set is unique!'

ax.text(1.02, 0.5, text_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Ablation coloring legend
ax2 = fig.add_subplot(1, 2, 2)
ax2.axis('off')
legend_text = "Why you see 'repetition':\n\n"
legend_text += "The 'repetition' is actually:\n"
legend_text += '• temp_30 at different parameters: h=40,ch=6 | h=40,ch=8 | h=40,ch=10 | h=30,ch=10 | h=30,ch=6 | h=30,ch=8\n'
legend_text += '• baseline at different parameters: similar parameter grid\n'
legend_text += '• no_velocity at different parameters: smaller subset\n\n'
legend_text += 'Each is a UNIQUE experiment because:\n'
legend_text += '✓ Different horizons → Different MPC planning windows\n'
legend_text += '✓ Different cost_horizons → Different robustification horizons\n'
legend_text += '✓ Different max_w → Different disturbance levels\n\n'
legend_text += "HOWEVER, the parameters don't matter!\n"
legend_text += '⚠️  All parameter combinations give similar violation rates\n'
legend_text += '🚨 This suggests the parameters are broken/unused'

ax2.text(0.05, 0.95, legend_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'investigation_repetition_explained.png'), dpi=300, bbox_inches='tight')
print('✓ Saved: investigation_repetition_explained.png')
plt.close()

print('\n' + '=' * 80)
print('✓ ALL INVESTIGATION PLOTS GENERATED')
print('=' * 80)
