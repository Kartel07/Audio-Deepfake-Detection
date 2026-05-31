import matplotlib.pyplot as plt

labels = ['In-Domain (Clean Data)', 'Out-of-Domain (Real World)']
eer_values = [0.35, 58.85]
colors = ['#2ca02c', '#d62728'] # Green for good, Red for bad

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(labels, eer_values, color=colors, width=0.5)

# Add the text directly on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Equal Error Rate (EER %)', fontsize=14, fontweight='bold')
ax.set_title('Version 1.0: The Generalization Collapse', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, 70)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("v1_collapse_chart.png", dpi=300)
plt.show()