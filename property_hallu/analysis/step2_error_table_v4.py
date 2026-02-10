import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------- refusal patterns (you can extend) ----------
REFUSAL_PATTERNS = [
    r"\bI (can't|cannot) (answer|provide)\b",
    r"\bI do not have (access|information)\b",
    r"\bI'?m sorry\b",
    r"\bnot available\b",
    r"\bunknown\b",
    r"\bno (information|data)\b",
    r"\bAs an AI\b",
    r"\bI cannot browse\b",
    r"\bI can'?t access\b",
]

def compute_refusal(raw: str) -> bool:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return False
    s = str(raw)
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, s, flags=re.I):
            return True
    return False

def norm_task(x: str) -> str:
    s = str(x).strip()
    # keep your two tasks normalized
    if s in ["band_gap", "bandgap", "band gap"]:
        return "band_gap"
    if s in ["formation_energy_per_atom", "formation energy per atom", "formation_energy"]:
        return "formation_energy_per_atom"
    return s

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def label_mode_guess(series_pred: pd.Series, top_k=3):
    """Return a set of mode values (float) used for mode_guess labeling."""
    vals = pd.to_numeric(series_pred, errors="coerce").dropna()
    if len(vals) == 0:
        return set()
    vc = vals.value_counts()
    modes = vc.head(top_k).index.tolist()
    return set([float(v) for v in modes])

def is_scale_error(gt, pred):
    """Heuristic: detect ~10x / 100x magnitude mismatch (common unit/scale mistakes)."""
    if gt is None or pred is None:
        return False
    ag, ap = abs(gt), abs(pred)
    if ag == 0 or ap == 0:
        return False
    ratio = ap / ag
    # near 10x or 0.1x or 100x or 0.01x
    for r in [10, 0.1, 100, 0.01]:
        if abs(ratio - r) / r < 0.15:  # 15% relative window
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="master_experiment_results_v4.csv")
    ap.add_argument("--out", default="results_v4", help="output dir")
    ap.add_argument("--focus_exp_contains", default=None,
                    help="Optional: only include experiments whose name contains this substring (e.g., 'baseline').")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.master)

    # --- map v4 columns to standard names ---
    # your v4 columns:
    # ['mp_id','property','ground_truth','raw_output','model','experiment','extracted_value','abs_error','is_correct']
    rename_map = {
        "mp_id": "material_id",
        "property": "task",
        "ground_truth": "gt_value",
        "experiment": "exp",
        "extracted_value": "pred_value",
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    # normalize
    df["task"] = df["task"].apply(norm_task)
    df["gt_value"] = df["gt_value"].apply(safe_float)
    df["pred_value"] = df["pred_value"].apply(safe_float)

    if args.focus_exp_contains:
        df = df[df["exp"].astype(str).str.contains(args.focus_exp_contains, case=False, na=False)].copy()

    # compute refusal + parse_fail
    df["refusal"] = df["raw_output"].apply(compute_refusal) if "raw_output" in df.columns else False
    df["parse_fail"] = df["pred_value"].isna()

    # determine error rows
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(df["is_correct"])
        df["is_correct"] = df["is_correct"].astype(bool)
    else:
        df["is_correct"] = False

    df["is_error"] = (~df["is_correct"]) | df["parse_fail"] | df["refusal"]

    # precompute mode sets per (task, model, exp)
    mode_map = {}
    for (task, model, exp), sub in df.groupby(["task","model","exp"], dropna=False):
        mode_map[(task, model, exp)] = label_mode_guess(sub["pred_value"], top_k=3)

    # assign auto error types
    err_types = []
    for _, r in df.iterrows():
        if not r["is_error"]:
            err_types.append("ok")
            continue

        if r["refusal"]:
            err_types.append("refusal")
            continue
        if r["parse_fail"]:
            err_types.append("parse_fail")
            continue

        task = r["task"]
        model = r["model"]
        exp = r["exp"]
        gt = r["gt_value"]
        pred = r["pred_value"]

        labels = []

        # mode guess
        modes = mode_map.get((task, model, exp), set())
        if pred is not None and float(pred) in modes:
            labels.append("mode_guess")

        if task == "formation_energy_per_atom":
            if gt is not None and pred is not None:
                if gt * pred < 0:
                    labels.append("sign_error")
                if is_scale_error(gt, pred):
                    labels.append("scale_error")

        if len(labels) == 0:
            labels.append("wrong_value_other")

        # combine
        err_types.append(", ".join(sorted(set(labels))))

    df["error_type_auto"] = err_types

    # save full annotated table (useful for inspection)
    out_full = outdir / "master_with_error_type_auto.csv"
    df.to_csv(out_full, index=False)
    print("Saved:", out_full)

    # distribution among errors (exclude ok)
    err = df[df["error_type_auto"] != "ok"].copy()

    dist = (
        err.groupby(["task","model","exp","error_type_auto"], dropna=False)
           .size()
           .reset_index(name="n")
    )
    dist["pct_in_group"] = dist["n"] / dist.groupby(["task","model","exp"])["n"].transform("sum")
    dist = dist.sort_values(["task","model","exp","pct_in_group"], ascending=[True,True,True,False]).reset_index(drop=True)

    out_dist = outdir / "error_type_distribution_v4.csv"
    dist.to_csv(out_dist, index=False)
    print("Saved:", out_dist)

    # pivot table for paper "big table"
    pivot = dist.pivot_table(
        index=["task","error_type_auto"],
        columns=["model","exp"],
        values="pct_in_group",
        aggfunc="first",
        fill_value=0.0,
    ).reset_index()

    out_pivot = outdir / "error_type_pivot_v4.csv"
    pivot.to_csv(out_pivot, index=False)
    print("Saved:", out_pivot)

    # quick print
    print("\nTop error types per (task, model, exp):")
    print(dist.groupby(["task","model","exp"]).head(3).to_string(index=False))

if __name__ == "__main__":
    main()
    