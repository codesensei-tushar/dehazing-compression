import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create figures directory
os.makedirs('results/figures', exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Pareto / Scatter Plot: Parameters vs. PSNR
teams = ['DeHamer (Teacher)', 'Node A (w16, GT)', 'Node B (w32, GT)', 'Node C (w32, Pseudo)']
params_m = [132.45, 4.35, 17.11, 17.11]
psnrs = [36.576, 32.39, 34.40, 33.87]

plt.figure(figsize=(8, 6))
# Create scatter
sns.scatterplot(x=params_m, y=psnrs, s=150, color="blue", marker='o')

# Annotations
for i, team in enumerate(teams):
    offset_x = 2
    offset_y = 0.1
    if "Node C" in team:
        offset_y = -0.2
    
    plt.annotate(team, 
                 (params_m[i], psnrs[i]), 
                 textcoords="offset points", 
                 xytext=(offset_x, offset_y*30), 
                 ha='left', fontsize=11)

plt.xscale('log')
plt.xlabel('Parameters (Millions, Log Scale)', fontsize=12)
plt.ylabel('PSNR (dB) on SOTS-Indoor', fontsize=12)
plt.title('Capacity vs Quality (Pareto Plot)', fontsize=14, fontweight='bold')
plt.xlim(2, 200)
plt.ylim(31.5, 37.5)
plt.tight_layout()
plt.savefig('results/figures/capacity_vs_quality_pareto.png', dpi=300)
plt.close()

# 2. Phase 1: PTQ Bar Chart
# Dynamic INT8 vs Mixed Precision vs Block Static
variants = ['FP32 (Ref)', 'INT8 Mixed', 'INT8 Dynamic', 'Block CNN Static']
psnr_ptq = [36.576, 36.551, 36.470, 34.545]
colors = ['#2ecc71', '#28b463', '#f1c40f', '#e74c3c']

plt.figure(figsize=(9, 5))
bars = plt.bar(variants, psnr_ptq, color=colors, width=0.6)
plt.ylim(33.5, 37)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.title('Phase 1: Post-Training Quantization PSNR Drops', fontsize=14, fontweight='bold')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.3f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('results/figures/ptq_psnr_drops.png', dpi=300)
plt.close()


# 3. Sensitivity Bar Chart
with open('results/dehamer_sensitivity_indoor.json') as f:
    sens_data = json.load(f)

# Sort by impact
items = sorted(sens_data.items(), key=lambda x: x[1], reverse=True)[:10]  # top 10
labels = [x[0].replace('swin_1.', '') for x in items]
values = [x[1] for x in items]

plt.figure(figsize=(10, 6))
sns.barplot(x=values, y=labels, palette="rocket")
plt.xlabel('PSNR Recovery (dB) when kept in FP32', fontsize=12)
plt.title('Top 10 Most Sensitive Linear Layers in DeHamer Swin Branch', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/sensitivity_bar_chart.png', dpi=300)
plt.close()

print("Plots successfully generated and saved to results/figures/")
