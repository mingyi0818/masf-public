import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

BASE_DIR = r"d:\ResearchPaperPrepare\masf-public"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load results
with open(os.path.join(RESULTS_DIR, "all_experiments_results.json"), 'r') as f:
    results = json.load(f)

# ============================================================
# Figure 1: MASF Architecture Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(5, 7.5, 'MASF Framework Architecture', fontsize=16, fontweight='bold', ha='center')

# Input box
ax.add_patch(plt.Rectangle((0.5, 6), 2, 0.8, facecolor='#E8F4FD', edgecolor='#2E86AB', linewidth=2))
ax.text(1.5, 6.4, 'Student Code\nSubmission', ha='center', va='center', fontsize=10)

# Misconception Detection
ax.add_patch(plt.Rectangle((3.5, 6), 3, 0.8, facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2))
ax.text(5, 6.4, 'Misconception Detection\n(KC Classification)', ha='center', va='center', fontsize=10, fontweight='bold')

# KC Binding
ax.add_patch(plt.Rectangle((7.5, 6), 2, 0.8, facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2))
ax.text(8.5, 6.4, 'KC Binding\n& Matching', ha='center', va='center', fontsize=10)

# Arrows
ax.annotate('', xy=(3.5, 6.4), xytext=(2.5, 6.4), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.annotate('', xy=(7.5, 6.4), xytext=(6.5, 6.4), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Scaffolded Feedback Layers
layers = [
    ('L1: Conceptual Scaffold', '#FFEBEE', '#C62828', 4.5),
    ('L2: Localized Hint', '#E3F2FD', '#1565C0', 3.0),
    ('L3: Guided Plan', '#F3E5F5', '#6A1B9A', 1.5)
]

for label, facecolor, edgecolor, y in layers:
    ax.add_patch(plt.Rectangle((3.5, y), 3, 0.8, facecolor=facecolor, edgecolor=edgecolor, linewidth=2))
    ax.text(5, y+0.4, label, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(5, y+0.8), xytext=(5, y+1.2 if y > 2 else y+0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Output
ax.add_patch(plt.Rectangle((7.5, 3), 2, 0.8, facecolor='#ECEFF1', edgecolor='#455A64', linewidth=2))
ax.text(8.5, 3.4, 'Adaptive\nFeedback', ha='center', va='center', fontsize=10)
ax.annotate('', xy=(7.5, 3.4), xytext=(6.5, 3.4), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Knowledge Component Bank
ax.add_patch(plt.Rectangle((0.5, 1.5), 2, 3.5, facecolor='#FFFDE7', edgecolor='#F9A825', linewidth=2, linestyle='--'))
ax.text(1.5, 4.6, 'KC Bank\n(67 Misconceptions)', ha='center', va='center', fontsize=9)
ax.text(1.5, 3.8, 'Control Flow\nVariable Scope\nFunction & Recursion\nString Operations\nList & Collection\nBoolean Logic\n...', ha='center', va='center', fontsize=7)

ax.annotate('', xy=(3.5, 5.5), xytext=(2.5, 4.0), arrowprops=dict(arrowstyle='->', color='gray', lw=1, linestyle='--'))
ax.annotate('', xy=(3.5, 3.5), xytext=(2.5, 3.0), arrowprops=dict(arrowstyle='->', color='gray', lw=1, linestyle='--'))

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'figure1_masf_architecture.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 1")

# ============================================================
# Figure 2: Classification Benchmark Results
# ============================================================
e1 = results['e1_classification']
methods = list(e1.keys())
metrics = ['accuracy_mean', 'f1_macro_mean', 'f1_micro_mean']
metric_labels = ['Accuracy', 'F1-Macro', 'F1-Micro']

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2E86AB', '#A23B72', '#F18F01']

for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
    values = [e1[m][metric] for m in methods]
    ax.bar(x + i*width, values, width, label=label, color=color, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('E1: KC-aware Misconception Classification Benchmark', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace(' + ', '\n+ ') for m in methods], fontsize=9)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'figure2_classification_benchmark.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 2")

# ============================================================
# Figure 3: Ablation Study Results
# ============================================================
e4 = results['e4_ablation']
variants = list(e4.keys())
acc_values = [e4[v]['accuracy_mean'] for v in variants]
f1_values = [e4[v]['f1_macro_mean'] for v in variants]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(variants))
width = 0.35

bars1 = ax.bar(x - width/2, acc_values, width, label='Accuracy', color='#2E86AB', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, f1_values, width, label='F1-Macro', color='#A23B72', edgecolor='black', linewidth=0.5)

ax.set_xlabel('Ablation Variant', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('E4: Ablation Study on Feature Components', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=9, rotation=15, ha='right')
ax.legend()
ax.set_ylim(0, 0.8)
ax.grid(axis='y', alpha=0.3)

# Highlight best
best_idx = np.argmax(f1_values)
ax.annotate('Best', xy=(best_idx, f1_values[best_idx]), xytext=(best_idx, f1_values[best_idx]+0.05),
            arrowprops=dict(arrowstyle='->', color='red'), color='red', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'figure3_ablation_study.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 3")

# ============================================================
# Figure 4: Parameter Sensitivity
# ============================================================
sens = pd.DataFrame(results['e5_sensitivity'])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# C parameter
sens_c = sens[sens['parameter'] == 'C']
axes[0].plot(sens_c['value'], sens_c['f1_macro'], marker='o', color='#2E86AB', linewidth=2, markersize=8)
axes[0].set_xlabel('C (Regularization)', fontsize=11)
axes[0].set_ylabel('F1-Macro', fontsize=11)
axes[0].set_title('Sensitivity to C', fontsize=12, fontweight='bold')
axes[0].set_xscale('log')
axes[0].grid(alpha=0.3)
axes[0].axvline(x=10.0, color='red', linestyle='--', alpha=0.5, label='Optimal')
axes[0].legend()

# max_features
sens_mf = sens[sens['parameter'] == 'max_features']
axes[1].plot(sens_mf['value'], sens_mf['f1_macro'], marker='s', color='#A23B72', linewidth=2, markersize=8)
axes[1].set_xlabel('Max Features', fontsize=11)
axes[1].set_ylabel('F1-Macro', fontsize=11)
axes[1].set_title('Sensitivity to Max Features', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)
axes[1].axvline(x=2000, color='red', linestyle='--', alpha=0.5, label='Optimal')
axes[1].legend()

# ngram_range
sens_ng = sens[sens['parameter'] == 'ngram_range']
axes[2].bar(range(len(sens_ng)), sens_ng['f1_macro'], color='#F18F01', edgecolor='black', linewidth=0.5)
axes[2].set_xlabel('N-gram Range', fontsize=11)
axes[2].set_ylabel('F1-Macro', fontsize=11)
axes[2].set_title('Sensitivity to N-gram Range', fontsize=12, fontweight='bold')
axes[2].set_xticks(range(len(sens_ng)))
axes[2].set_xticklabels(sens_ng['value'], fontsize=9, rotation=15)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'figure4_parameter_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 4")

# ============================================================
# Figure 5: KC Distribution (Pie Chart)
# ============================================================
kc_dist = results['data_stats']['kc_distribution']
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(kc_dist)))
wedges, texts, autotexts = ax.pie(kc_dist.values(), labels=kc_dist.keys(), autopct='%1.1f%%',
                                   colors=colors, startangle=90, textprops={'fontsize': 9})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax.set_title('Distribution of Knowledge Components in McMiner Dataset', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'figure5_kc_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved Figure 5")

print("\nAll figures generated successfully!")
