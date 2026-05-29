import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os

# Configuration
export_dir = "Paper1_Data_Exports"
train_env = "FoR"
test_env = "ASVspoof5"

# Define the models and their aesthetic properties for the plot
# Thickened lines for better visibility
models = {
    "V1": {"name": "V1 (CNN Baseline)", "color": "#d62728", "linestyle": ":", "linewidth": 3},
    "V2": {"name": "V2 (Deep ResNet)", "color": "#ff7f0e", "linestyle": "--", "linewidth": 3},
    "V4": {"name": "V4 (RawNet2)", "color": "#8c564b", "linestyle": "-.", "linewidth": 3},
    "V3": {"name": "V3 (XLS-R Transformer)", "color": "#2ca02c", "linestyle": "-", "linewidth": 3},
    "V5": {"name": "V5 (Orthogonal Fusion)", "color": "#1f77b4", "linestyle": "-", "linewidth": 4.5} # Extra thick for emphasis
}

# 1. Increase Figure Size and DPI for high-resolution rendering
plt.style.use('seaborn-v0_8-paper')
# Changed from (8,8) to (10, 8) to give the legend more breathing room horizontally
fig, ax = plt.subplots(figsize=(10, 8))

print("📈 Generating High-Readability Cross-Corpus DET Curves...")

for v_key, props in models.items():
    csv_filename = f"raw_scores_{v_key}_Train{train_env}_Test{test_env}.csv"
    csv_path = os.path.join(export_dir, csv_filename)
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        labels = df['True_Label'].values
        scores = df['Predicted_Score'].values
        
        # Calculate standard ROC metrics
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr # False Rejection Rate
        
        # Plotting FAR (fpr) vs FRR (fnr)
        ax.plot(fpr * 100, fnr * 100, label=props["name"], 
                color=props["color"], linestyle=props["linestyle"], linewidth=props["linewidth"])
        
        print(f"✅ Plotted {props['name']}")
    else:
        print(f"⚠️ Missing {csv_filename} - Skipping.")

# Formatting the DET Curve
ax.set_xscale('log')
ax.set_yscale('log')

# Standardizing axis limits for DET curves
ax.set_xlim([0.1, 100])
ax.set_ylim([0.1, 100])

# Setting specific tick marks for logarithmic readability
ticks = [0.1, 1, 5, 10, 20, 40, 100]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

# 2. Drastically increase Font Sizes for readability
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('False Acceptance Rate (FAR %)', fontsize=16, fontweight='bold')
ax.set_ylabel('False Rejection Rate (FRR %)', fontsize=16, fontweight='bold')
ax.set_title(f'Cross-Corpus DET Analysis\n(Train: {train_env} $\\rightarrow$ Test: {test_env})', 
             fontsize=18, fontweight='bold', pad=20)

ax.grid(True, which="both", ls="--", alpha=0.5)

# 3. Optimize Legend Placement and Size
ax.legend(loc='lower left', fontsize=14, frameon=True, shadow=True, borderpad=1)

# Plot the diagonal Random Chance line for reference
ax.plot([0.1, 100], [0.1, 100], color='black', linestyle=':', alpha=0.6, linewidth=2, label='Random Chance')

plt.tight_layout()
output_path = "det_curves.png"
# Increased DPI from 300 to 400 for even crisper lines in LaTeX
plt.savefig(output_path, format='png', dpi=400, bbox_inches='tight')
print(f"🎉 High-Res DET Curve successfully saved to '{output_path}'")
plt.show()