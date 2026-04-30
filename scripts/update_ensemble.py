import json

with open(r'D:\Datathon-2026\data\baseline_clean.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][30]
new_source = []
for line in cell['source']:
    # Replace ALL et references with xgb
    line = line.replace('et_val_pred', 'xgb_val_pred')
    line = line.replace('test_pred_et', 'test_pred_xgb')
    line = line.replace('"et": ', '"xgb": ')
    line = line.replace('"et"', '"xgb"')
    line = line.replace('et_val_df', 'xgb_val_df')
    line = line.replace('et_test_df', 'xgb_test_df')
    new_source.append(line)

cell['source'] = new_source

with open(r'D:\Datathon-2026\data\baseline_clean.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed remaining ET references")

# Verify
with open(r'D:\Datathon-2026\data\baseline_clean.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cell = nb['cells'][30]
src = ''.join(cell['source'])
print("Contains xgb_val_pred:", "xgb_val_pred" in src)
print("Contains et_val_pred:", "et_val_pred" in src)
print("Contains xgb_test_df:", "xgb_test_df" in src)
print("Contains et_test_df:", "et_test_df" in src)
print("Contains \"xgb\":", '\"xgb\"' in src)
print("Contains \"et\":", '\"et\"' in src)
