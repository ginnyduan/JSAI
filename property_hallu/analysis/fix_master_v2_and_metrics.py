import pandas as pd
import numpy as np
from pathlib import Path
import re

MASTER = "/Users/yutongduan/llamp/experiments/property_hallu/analysis/master_experiment_results_v2.csv"
GT200  = "/Users/yutongduan/llamp/experiments/property_hallu/results/gt_200_full.csv"

OUT_DIR = Path("/Users/yutongduan/llamp/experiments/property_hallu/analysis/results_v2_fixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1) load ---
m = pd.read_csv(MASTER)
gt = pd.read_csv(GT200)

# --- 2) fix property name ---
m["property"] = m["property"].replace({
    "formation_energy_per_atom_per_atom": "formation_energy_per_atom"
})

# --- 3) build long GT table (mp_id, property, gt_value_from_gt200) ---
# gt_200_full.csv should contain: material_id, band_gap, formation_energy_per_atom, ...
gt_long = gt[["material_id", "band_gap", "formation_energy_per_atom"]].copy()
gt_long = gt_long.rename(columns={"material_id": "mp_id"})
gt_long = gt_long.melt(
    id_vars=["mp_id"],
    value_vars=["band_gap", "formation_energy_per_atom"],
    var_name="property",
    value_name="gt_value_from_gt200"
)

# --- 4) merge to fill gt_value ---
m = m.merge(gt_long, on=["mp_id", "property"], how="left")

# If gt_value missing in master, fill from gt_200_full
if "gt_value" not in m.columns:
    m["gt_value"] = np.nan
m["gt_value"] = m["gt_value"].fillna(m["gt_value_from_gt200"])

# tool_value：如果你想统一为“工具返回值”，而 v2 里很多实验没有 tool_value，也可以先填成 gt_value（对 baseline 无所谓）
if "tool_value" not in m.columns:
    m["tool_value"] = np.nan
m["tool_value"] = m["tool_value"].fillna(m["gt_value"])

m = m.drop(columns=["gt_value_from_gt200"])

# --- 5) recompute abs_error & is_correct with per-property tolerances ---
TOL = {
    "band_gap": 0.05,
    "formation_energy_per_atom": 0.05
}

# abs_error only meaningful if extracted_value and gt_value exist
m["abs_error"] = np.where(
    m["extracted_value"].notna() & m["gt_value"].notna(),
    (m["extracted_value"] - m["gt_value"]).abs(),
    np.nan
)

def is_correct_row(row):
    prop = row["property"]
    tol = TOL.get(prop, 0.05)
    if pd.isna(row["abs_error"]):
        return np.nan
    return row["abs_error"] <= tol

m["is_correct"] = m.apply(is_correct_row, axis=1)

# answered/parse_fail/refusal：保持你已有列（如果没有就推断）
# 这里不强行改你的定义，只做一个“更强 refusal regex”可选项：
REFUSAL_PAT = re.compile(
    r"(i\s+can't\s+access|cannot\s+access|don't\s+have\s+access|unable\s+to\s+access|"
    r"i\s+cannot\s+browse|can't\s+browse|no\s+internet|do\s+not\s+have\s+the\s+ability|"
    r"as\s+an\s+ai\s+language\s+model|not\s+available\s+to\s+me|i\s+do\s+not\s+have\s+the\s+data)",
    re.IGNORECASE
)
if "refusal" in m.columns:
    m["refusal"] = m["refusal"].fillna(False)
    # 强化：如果 raw_output 命中拒答模式 -> refusal=True
    m.loc[m["raw_output"].astype(str).str.contains(REFUSAL_PAT, na=False), "refusal"] = True

if "answered" not in m.columns:
    m["answered"] = m["extracted_value"].notna()

if "parse_fail" not in m.columns:
    m["parse_fail"] = m["extracted_value"].isna()

# --- 6) recompute metrics (Table A) ---
def agg(g):
    n = len(g)
    answer_rate = g["answered"].mean()
    parse_fail_rate = g["parse_fail"].mean()
    refusal_rate = g["refusal"].mean() if "refusal" in g.columns else np.nan

    overall_acc = g["is_correct"].mean(skipna=True)

    if (g["answered"] == True).any():
        acc_given_answered = g.loc[g["answered"] == True, "is_correct"].mean(skipna=True)
    else:
        acc_given_answered = np.nan

    return pd.Series({
        "n": n,
        "answer_rate": answer_rate,
        "parse_fail_rate": parse_fail_rate,
        "refusal_rate": refusal_rate,
        "overall_acc": overall_acc,
        "acc_given_answered": acc_given_answered
    })

metrics = (
    m.groupby(["model","experiment","property"], dropna=False)
     .apply(agg)
     .reset_index()
     .sort_values(["model","experiment","property"])
)

# --- 7) save ---
master_out  = OUT_DIR / "master_experiment_results_v2_fixed.csv"
metrics_out = OUT_DIR / "metrics_by_model_exp_task_v2_fixed.csv"
m.to_csv(master_out, index=False)
metrics.to_csv(metrics_out, index=False)

print("Saved:", master_out)
print("Saved:", metrics_out)
print(metrics)