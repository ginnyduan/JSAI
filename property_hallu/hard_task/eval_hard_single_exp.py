import argparse
import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_KEYS = [
    "material_id",
    "band_gap_eV",
    "formation_energy_per_atom_eV",
    "is_metal",
    "energy_above_hull_eV",
    "is_stable",
]

TOL = {
    "band_gap_eV": 1e-6,
    "formation_energy_per_atom_eV": 1e-6,
    "energy_above_hull_eV": 1e-6,
}

REFUSAL_PATTERNS = [
    r"\bI (can't|cannot)\b",
    r"\bI do not have\b",
    r"\bI don't have\b",
    r"\bI'm sorry\b",
    r"\bnot available\b",
    r"\bunknown\b",
    r"\bno (information|data)\b",
    r"\bAs an AI\b",
    r"\bI cannot browse\b",
    r"\bI can'?t access\b",
    r"\bunable to\b",
    r"\bcannot provide\b",
]

def compute_refusal(raw: str) -> bool:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return False
    s = str(raw)
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, s, flags=re.I):
            return True
    return False

def safe_json_load(s: str):
    if s is None:
        return None
    txt = str(s).strip()
    # strip code fences
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt.replace("json\n", "").replace("JSON\n", "")
    l = txt.find("{")
    r = txt.rfind("}")
    if l >= 0 and r >= 0 and r > l:
        txt = txt[l:r+1]
    try:
        return json.loads(txt)
    except:
        return None

def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))

def close(a, b, tol):
    if a is None or b is None:
        return False
    if not is_number(a) or not is_number(b):
        return False
    return abs(float(a) - float(b)) <= tol

def check_constraints(obj: dict) -> bool:
    try:
        if obj.get("is_metal") is True:
            bg = obj.get("band_gap_eV")
            if not is_number(bg) or float(bg) > 0.1:
                return False
        if obj.get("is_stable") is True:
            eah = obj.get("energy_above_hull_eV")
            if not is_number(eah) or float(eah) > 1e-6:
                return False
    except:
        return False
    return True

def build_gt_long(gt_df: pd.DataFrame) -> pd.DataFrame:
    need = ["material_id","band_gap","formation_energy_per_atom","is_metal","energy_above_hull","is_stable"]
    miss = [c for c in need if c not in gt_df.columns]
    if miss:
        raise ValueError(f"GT missing cols={miss}. got={gt_df.columns.tolist()}")

    rows = []
    for _, r in gt_df.iterrows():
        tool = {
            "material_id": str(r["material_id"]),
            "band_gap": None if pd.isna(r["band_gap"]) else float(r["band_gap"]),
            "formation_energy_per_atom": None if pd.isna(r["formation_energy_per_atom"]) else float(r["formation_energy_per_atom"]),
            "is_metal": None if pd.isna(r["is_metal"]) else bool(r["is_metal"]),
            "energy_above_hull": None if pd.isna(r["energy_above_hull"]) else float(r["energy_above_hull"]),
            "is_stable": None if pd.isna(r["is_stable"]) else bool(r["is_stable"]),
        }
        std = {
            "material_id": tool["material_id"],
            "band_gap_eV": tool["band_gap"],
            "formation_energy_per_atom_eV": tool["formation_energy_per_atom"],
            "is_metal": tool["is_metal"],
            "energy_above_hull_eV": tool["energy_above_hull"],
            "is_stable": tool["is_stable"],
        }
        rows.append({"mp_id": tool["material_id"], "gt_json": json.dumps(std, ensure_ascii=False)})
    return pd.DataFrame(rows)

def eval_row(raw_output, gt_obj):
    refusal = compute_refusal(raw_output)
    if refusal:
        return {
            "refusal": True,
            "parse_fail": False,
            "schema_fail": False,
            "value_mismatch": False,
            "constraint_violation": False,
            "is_correct": False,
        }

    pred = safe_json_load(raw_output)
    if not isinstance(pred, dict):
        return {
            "refusal": False,
            "parse_fail": True,
            "schema_fail": False,
            "value_mismatch": False,
            "constraint_violation": False,
            "is_correct": False,
        }

    # schema
    schema_fail = any(k not in pred for k in REQUIRED_KEYS)
    if schema_fail:
        return {
            "refusal": False,
            "parse_fail": False,
            "schema_fail": True,
            "value_mismatch": True,
            "constraint_violation": True,
            "is_correct": False,
        }

    # type/value checks
    mism = False

    # material_id exact
    if str(pred["material_id"]) != str(gt_obj["material_id"]):
        mism = True

    # booleans
    for k in ["is_metal","is_stable"]:
        if not isinstance(pred[k], bool):
            mism = True
        elif pred[k] != gt_obj[k]:
            mism = True

    # numerics
    for k in ["band_gap_eV","formation_energy_per_atom_eV","energy_above_hull_eV"]:
        if not is_number(pred[k]):
            mism = True
        elif not close(pred[k], gt_obj[k], TOL[k]):
            mism = True

    constraint_violation = (not check_constraints(pred))

    is_correct = (not mism) and (not constraint_violation)
    return {
        "refusal": False,
        "parse_fail": False,
        "schema_fail": False,
        "value_mismatch": mism,
        "constraint_violation": constraint_violation,
        "is_correct": is_correct,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="gt_200_full.csv")
    ap.add_argument("--pred", required=True, help="one experiment csv (mp_id, raw_output, model, experiment, ...)")
    ap.add_argument("--outdir", default="eval_out", help="output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gt_df = pd.read_csv(args.gt)
    gt_long = build_gt_long(gt_df)

    pred_df = pd.read_csv(args.pred)
    need = ["mp_id","raw_output"]
    miss = [c for c in need if c not in pred_df.columns]
    if miss:
        raise ValueError(f"Pred file missing cols={miss}. got={pred_df.columns.tolist()}")

    pred_df["mp_id"] = pred_df["mp_id"].astype(str)
    merged = pred_df.merge(gt_long, on="mp_id", how="left")
    if merged["gt_json"].isna().any():
        bad = merged[merged["gt_json"].isna()][["mp_id"]].head(20)
        raise ValueError(f"Some mp_id not found in GT. examples:\n{bad.to_string(index=False)}")

    # eval per row
    scores = []
    for _, r in merged.iterrows():
        gt_obj = json.loads(r["gt_json"])
        scores.append(eval_row(r["raw_output"], gt_obj))
    score_df = pd.DataFrame(scores)
    merged = pd.concat([merged, score_df], axis=1)

    # metrics
    n = len(merged)
    answer_rate = 1.0 - merged["refusal"].mean()
    refusal_rate = merged["refusal"].mean()
    parse_fail_rate = merged["parse_fail"].mean()
    schema_fail_rate = merged["schema_fail"].mean()
    constraint_violation_rate = merged["constraint_violation"].mean()
    overall_acc = merged["is_correct"].mean()
    acc_given_answered = merged.loc[~merged["refusal"], "is_correct"].mean() if (~merged["refusal"]).sum() else np.nan

    model = merged["model"].iloc[0] if "model" in merged.columns and len(merged) else ""
    exp = merged["experiment"].iloc[0] if "experiment" in merged.columns and len(merged) else Path(args.pred).stem

    metrics = pd.DataFrame([{
        "model": model,
        "experiment": exp,
        "n": n,
        "answer_rate": float(answer_rate),
        "refusal_rate": float(refusal_rate),
        "parse_fail_rate": float(parse_fail_rate),
        "schema_fail_rate": float(schema_fail_rate),
        "constraint_violation_rate": float(constraint_violation_rate),
        "overall_acc": float(overall_acc),
        "acc_given_answered": float(acc_given_answered) if not np.isnan(acc_given_answered) else np.nan,
    }])

    scored_path = outdir / f"scored_{Path(args.pred).stem}.csv"
    metrics_path = outdir / f"metrics_{Path(args.pred).stem}.csv"
    merged.to_csv(scored_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    print("Saved scored:", scored_path)
    print("Saved metrics:", metrics_path)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()