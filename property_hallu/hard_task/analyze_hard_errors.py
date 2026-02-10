import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Hard-task expected keys (your GT json has these)
REQ_KEYS = [
    "material_id",
    "band_gap_eV",
    "formation_energy_per_atom_eV",
    "is_metal",
    "energy_above_hull_eV",
    "is_stable",
]

NUM_KEYS = ["band_gap_eV", "formation_energy_per_atom_eV", "energy_above_hull_eV"]
BOOL_KEYS = ["is_metal", "is_stable"]

# ---------- JSON extraction ----------
def extract_json_obj(raw: str):
    """
    Extract a JSON dict from raw output that may be wrapped in ```json ... ``` or contain extra text.
    Returns dict or None.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()

    # strip common fences
    s = re.sub(r"^```json\s*", "", s, flags=re.I)
    s = re.sub(r"^```\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)

    # find first {...}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    blob = m.group(0)

    # try normal json
    try:
        return json.loads(blob)
    except:
        # handle doubled quotes like ""material_id""
        blob2 = blob.replace('""', '"')
        try:
            return json.loads(blob2)
        except:
            return None

def safe_loads(s: str):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        return json.loads(s)
    except:
        # handle doubled quotes
        try:
            return json.loads(str(s).replace('""','"'))
        except:
            return None

def to_float(x):
    if x is None: return None
    try: return float(x)
    except: return None

def to_bool(x):
    if x is None: return None
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    if s in ["true","1","yes","y"]: return True
    if s in ["false","0","no","n"]: return False
    return None

# ---------- constraint checks (same spirit as your evaluator) ----------
def constraint_violations(pred: dict):
    """
    Return list of constraint violation codes found in pred dict.
    You can adjust thresholds here.
    """
    v = []
    if not isinstance(pred, dict):
        return v

    bg = to_float(pred.get("band_gap_eV"))
    is_metal = to_bool(pred.get("is_metal"))
    eah = to_float(pred.get("energy_above_hull_eV"))
    is_stable = to_bool(pred.get("is_stable"))

    # C1: metal => band gap ~ 0
    if is_metal is True and bg is not None and bg > 0.1:
        v.append("metal_bg_nonzero")

    # C2: stable => energy_above_hull ~ 0
    if is_stable is True and eah is not None and abs(eah) > 1e-6:
        v.append("stable_eah_nonzero")

    # optional physical prior: formation_energy <= 0 (not strict; just flag)
    fe = to_float(pred.get("formation_energy_per_atom_eV"))
    if fe is not None and fe > 0:
        v.append("fe_positive_prior")

    return v

# ---------- reason classifier ----------
def classify_reason(row, tol_abs=0.05):
    """
    Produce a primary_reason + secondary_reason + details.
    We consider many cases:
    - refusal / parse_fail / schema_fail
    - material_id mismatch
    - missing keys
    - numeric rounding (tiny error) vs true mismatch
    - boolean mismatch
    - constraint violation
    - field swap suspicion
    """
    raw = row.get("raw_output", None)
    pred = row.get("_pred_obj", None)
    gt = row.get("_gt_obj", None)

    # 0) tool flags if already exist in scored CSV
    for flag in ["refusal", "parse_fail", "schema_fail"]:
        if flag in row and bool(row[flag]) is True:
            return flag, "", ""

    # 1) JSON parse
    if not isinstance(pred, dict):
        return "json_parse_fail", "", "cannot parse JSON"

    # 2) GT existence
    if not isinstance(gt, dict):
        return "gt_missing_or_bad", "", "cannot parse gt_json"

    # 3) missing required keys
    missing = [k for k in REQ_KEYS if k not in pred]
    if missing:
        return "missing_keys", "", f"missing={missing}"

    # 4) material_id mismatch (critical in hard task)
    pmid = str(pred.get("material_id"))
    gmid = str(gt.get("material_id"))
    if pmid != gmid:
        return "material_id_mismatch", "", f"pred={pmid} gt={gmid}"

    # 5) compute numeric errors
    num_errs = {}
    for k in NUM_KEYS:
        pv = to_float(pred.get(k))
        gv = to_float(gt.get(k))
        if pv is None or gv is None:
            num_errs[k] = None
        else:
            num_errs[k] = abs(pv - gv)

    # 6) boolean mismatches
    bool_mis = []
    for k in BOOL_KEYS:
        pv = to_bool(pred.get(k))
        gv = to_bool(gt.get(k))
        if pv is None or gv is None:
            continue
        if pv != gv:
            bool_mis.append(k)

    # 7) constraint violations (based on pred)
    cv = constraint_violations(pred)

    # 8) decide if it's "rounding"
    # rounding if all numeric errors <= 1e-3 and no bool mismatch
    numeric_vals = [e for e in num_errs.values() if e is not None]
    max_err = max(numeric_vals) if numeric_vals else None

    if max_err is not None and max_err <= 1e-3 and len(bool_mis) == 0:
        return "rounding_tiny_diff", "", f"max_err={max_err}"

    # 9) if boolean mismatch dominates
    if len(bool_mis) > 0 and (max_err is None or max_err <= tol_abs):
        return "boolean_mismatch", "", f"mismatch={bool_mis}; max_num_err={max_err}"

    # 10) if constraint violation exists
    if len(cv) > 0:
        # if also numeric mismatch, report both
        sec = "numeric_mismatch" if (max_err is not None and max_err > tol_abs) else ""
        return "constraint_violation", sec, f"violations={cv}; max_num_err={max_err}; bool_mis={bool_mis}"

    # 11) field swap suspicion: band_gap close to FE or vice versa
    # (rough heuristic: abs(pred_bg - gt_fe) small and abs(pred_fe - gt_bg) small)
    bg_p = to_float(pred.get("band_gap_eV"))
    fe_p = to_float(pred.get("formation_energy_per_atom_eV"))
    bg_g = to_float(gt.get("band_gap_eV"))
    fe_g = to_float(gt.get("formation_energy_per_atom_eV"))
    if None not in [bg_p, fe_p, bg_g, fe_g]:
        if abs(bg_p - fe_g) <= tol_abs and abs(fe_p - bg_g) <= tol_abs:
            return "field_swap_suspected", "", f"bg_p~fe_g and fe_p~bg_g (tol={tol_abs})"

    # 12) default: numeric mismatch / other
    if max_err is not None and max_err > tol_abs:
        return "numeric_mismatch", "", f"max_err={max_err}; per_field={num_errs}; bool_mis={bool_mis}"

    # if reached here but still incorrect, label as "other"
    return "other", "", f"max_err={max_err}; per_field={num_errs}; bool_mis={bool_mis}; cv={cv}"

def run_one(scored_path: Path, tol_abs=0.05, topk_examples=10):
    df = pd.read_csv(scored_path)
    # parse pred+gt
    df["_pred_obj"] = df["raw_output"].apply(extract_json_obj) if "raw_output" in df.columns else None
    df["_gt_obj"] = df["gt_json"].apply(safe_loads) if "gt_json" in df.columns else None

    # ensure is_correct exists
    if "is_correct" not in df.columns:
        raise ValueError("scored CSV must have is_correct column")

    wrong = df[df["is_correct"] == False].copy()
    acc = df["is_correct"].mean()

    print("=" * 80)
    print("FILE:", scored_path)
    print(f"n={len(df)} acc={acc:.4f} wrong={len(wrong)}")

    if len(wrong) == 0:
        print("No errors 🎉")
        return

    # classify
    reasons = wrong.apply(lambda r: classify_reason(r, tol_abs=tol_abs), axis=1)
    wrong["primary_reason"] = [x[0] for x in reasons]
    wrong["secondary_reason"] = [x[1] for x in reasons]
    wrong["details"] = [x[2] for x in reasons]

    # reason counts
    cnt = wrong["primary_reason"].value_counts().reset_index()
    cnt.columns = ["primary_reason", "count"]
    cnt["pct_of_wrong"] = cnt["count"] / len(wrong)

    print("\nTop error reasons:")
    print(cnt.to_string(index=False))

    # show a few examples per top reason
    print("\nExamples:")
    for reason in cnt["primary_reason"].head(6).tolist():
        sub = wrong[wrong["primary_reason"] == reason].head(topk_examples)
        cols = ["mp_id", "primary_reason", "secondary_reason", "details"]
        if "value_mismatch" in wrong.columns: cols.append("value_mismatch")
        if "constraint_violation" in wrong.columns: cols.append("constraint_violation")
        print("\n---", reason, f"(show {len(sub)}) ---")
        print(sub[cols].to_string(index=False))

    # save report
    out_report = scored_path.with_name(scored_path.stem + "_error_report.csv")
    wrong.to_csv(out_report, index=False)
    print("\nSaved detailed error report:", out_report)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True, help="One or more scored CSV paths")
    ap.add_argument("--tol_abs", type=float, default=0.05, help="abs tolerance for numeric correctness (paper setting)")
    ap.add_argument("--topk", type=int, default=10, help="examples per reason")
    args = ap.parse_args()

    for p in args.scored:
        run_one(Path(p), tol_abs=args.tol_abs, topk_examples=args.topk)

if __name__ == "__main__":
    main()