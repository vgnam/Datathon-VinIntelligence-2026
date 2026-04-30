import json

with open(r'D:\Datathon-2026\data\baseline_clean.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Add new cell after ensemble cell (index 30)
plot_code = '''import matplotlib.pyplot as plt

# Plot: Ensemble vs Baseline+ vs Actual (Validation)
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Revenue
axes[0].plot(val_dates, y_val_cmp["Revenue"], lw=1.2, label="Actual", color="black")
axes[0].plot(val_dates, baseline_val_df["Revenue"], lw=1.0, linestyle="--", label="Baseline+", color="blue")
axes[0].plot(val_dates, ens_val_rev, lw=1.0, linestyle="-", label="Ensemble", color="red")
axes[0].fill_between(val_dates, y_val_cmp["Revenue"], ens_val_rev, alpha=0.2, color="red", label="Ensemble Error")
axes[0].set_title("Revenue: Actual vs Baseline+ vs Ensemble (Validation)")
axes[0].legend(loc="upper left")
axes[0].set_ylabel("Revenue")

# COGS
axes[1].plot(val_dates, y_val_cmp["COGS"], lw=1.2, label="Actual", color="black")
axes[1].plot(val_dates, baseline_val_df["COGS"], lw=1.0, linestyle="--", label="Baseline+", color="blue")
axes[1].plot(val_dates, ens_val_cogs, lw=1.0, linestyle="-", label="Ensemble", color="red")
axes[1].fill_between(val_dates, y_val_cmp["COGS"], ens_val_cogs, alpha=0.2, color="red", label="Ensemble Error")
axes[1].set_title("COGS: Actual vs Baseline+ vs Ensemble (Validation)")
axes[1].legend(loc="upper left")
axes[1].set_ylabel("COGS")
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.show()

# Print metrics comparison
print("\\n=== VALIDATION METRICS COMPARISON ===")
print(f"Baseline+ MAPE Revenue: {mape_rev['base']:.4f}")
print(f"Ensemble  MAPE Revenue: {mape_np(y_val_cmp['Revenue'], ens_val_rev):.4f}")
print(f"Baseline+ MAPE COGS   : {mape_cogs['base']:.4f}")
print(f"Ensemble  MAPE COGS   : {mape_np(y_val_cmp['COGS'], ens_val_cogs):.4f}")
'''

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'ensemble-plot-001',
    'metadata': {},
    'outputs': [],
    'source': [line + '\n' for line in plot_code.split('\n') if line]
}

# Insert after cell 30
nb['cells'].insert(31, new_cell)

with open(r'D:\Datathon-2026\data\baseline_clean.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Added ensemble plot cell after ensemble section (cell 31)')
