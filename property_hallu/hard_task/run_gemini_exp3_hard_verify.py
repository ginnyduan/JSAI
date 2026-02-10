import os
import time
import json
import math
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any

# =========================
# CONFIG
# =========================
IN_CANDIDATE = Path("/Users/yutongduan/llamp/experiments/property_hallu/hard_task/results/results_hard_v1/gemini/gemini_exp2_hard_grounded_novfy.csv")
OUT_DIR      = Path("/Users/yutongduan/llamp/experiments/property_hallu/hard_task/results/results_hard_v1/gemini")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "exp3_hard_verify_override"

# Tolerances for numeric match (tool->pred)
TOL = {
    "band_gap_eV": 1e-6,
    "formation_energy_per_atom_eV": 1e-6,
    "energy_above_hull_eV": 1e-6,
}

def to_std_json(tool_record: dict) -> dict:
    """Map tool_record -> required schema JSON."""
    return {
        "material_id": str(tool_record["material_id"]),
        "band_gap_eV": float(tool_record["band_gap"]) if tool_record.get("band_gap") is not None else None,
        "formation_energy_per_atom_eV": float(tool_record["formation_energy_per_atom"]) if tool_record.get("formation_energy_per_atom") is not None else None,
        "is_metal": bool(tool_record["is_metal"]) if tool_record.get("is_metal") is not None else None,
        "energy_above_hull_eV": float(tool_record["energy_above_hull"]) if tool_record.get("energy_above_hull") is not None else None,
        "is_stable": bool(tool_record["is_stable"]) if tool_record.get("is_stable") is not None else None,
    }

def safe_json_load(s: str):
    """Try parse JSON even if model adds code fences."""
    if s is None:
        return None
    txt = str(s).strip()
    # strip ```json ... ```
    if txt.startswith("```"):
        txt = txt.strip("`")
        # after stripping backticks, might still include "json\n"
        txt = txt.replace("json\n", "").replace("JSON\n", "")
    # attempt locate first { ... last }
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
    # constraints:
    # - if is_metal True -> band_gap_eV <= 0.1
    # - if is_stable True -> energy_above_hull_eV <= 1e-6
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

def check_against_tool(pred: Dict[str, Any], tool_std: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Return ok, list of fail reasons."""
    reasons = []

    # schema keys
    req = ["material_id","band_gap_eV","formation_energy_per_atom_eV","is_metal","energy_above_hull_eV","is_stable"]
    for k in req:
        if k not in pred:
            reasons.append(f"missing_key:{k}")

    if reasons:
        return False, reasons

    # material_id exact
    if str(pred["material_id"]) != str(tool_std["material_id"]):
        reasons.append("material_id_mismatch")

    # booleans exact
    for k in ["is_metal","is_stable"]:
        if not isinstance(pred[k], bool):
            reasons.append(f"type_error:{k}")
        elif pred[k] != tool_std[k]:
            reasons.append(f"value_mismatch:{k}")

    # numerics tol
    for k in ["band_gap_eV","formation_energy_per_atom_eV","energy_above_hull_eV"]:
        if not is_number(pred[k]):
            reasons.append(f"type_error:{k}")
        elif not close(pred[k], tool_std[k], TOL[k]):
            reasons.append(f"value_mismatch:{k}")

    # constraints
    if not check_constraints(pred):
        reasons.append("constraint_violation")

    return (len(reasons) == 0), reasons

def main():
    df = pd.read_csv(IN_CANDIDATE)
    need = ["mp_id","raw_output","tool_record_json"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Candidate file missing cols={missing}. cols={df.columns.tolist()}")

    rows = []
    for _, r in df.iterrows():
        mpid = str(r["mp_id"])
        tool_record = json.loads(r["tool_record_json"])
        tool_std = to_std_json(tool_record)

        pred = safe_json_load(r["raw_output"])
        ok = False
        reasons = ["parse_fail"]
        if isinstance(pred, dict):
            ok, reasons = check_against_tool(pred, tool_std)

        final = pred if (ok and isinstance(pred, dict)) else tool_std  # override
        rows.append({
            "mp_id": mpid,
            "raw_output": json.dumps(final, ensure_ascii=False),
            "model": "gemini",
            "experiment": EXPERIMENT,
            "ts": time.time(),
            "override_applied": (not ok),
            "fail_reasons": ";".join(reasons) if reasons else "",
        })

    out_path = OUT_DIR / "gemini_exp3_hard_verify_override.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()