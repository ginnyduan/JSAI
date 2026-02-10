import os
import time
import json
import pandas as pd
from pathlib import Path
import google.generativeai as genai

# =========================
# CONFIG
# =========================
CSV_PATH = Path("/Users/yutongduan/llamp/experiments/property_hallu/results/gt_200_full.csv")
OUT_DIR  = Path("/Users/yutongduan/llamp/experiments/property_hallu/hard_task/results/results_hard_v1/gemini")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "gemini-2.5-flash-lite"
EXPERIMENT = "exp2_hard_grounded_novfy"
SLEEP_SEC  = 0.6

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY)")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def build_tool_record(row) -> dict:
    # tool record uses "tool units"
    return {
        "material_id": str(row["material_id"]),
        "band_gap": None if pd.isna(row["band_gap"]) else float(row["band_gap"]),  # eV
        "formation_energy_per_atom": None if pd.isna(row["formation_energy_per_atom"]) else float(row["formation_energy_per_atom"]),  # eV/atom
        "is_metal": None if pd.isna(row["is_metal"]) else bool(row["is_metal"]),
        "energy_above_hull": None if pd.isna(row["energy_above_hull"]) else float(row["energy_above_hull"]),  # eV
        "is_stable": None if pd.isna(row["is_stable"]) else bool(row["is_stable"]),
        "units": {
            "band_gap": "eV",
            "formation_energy_per_atom": "eV/atom",
            "energy_above_hull": "eV",
        }
    }

def build_prompt(mpid: str, tool_record: dict) -> str:
    tool_json = json.dumps(tool_record, ensure_ascii=False)
    return f"""You are a materials assistant.

User asks: "Return the following properties for {mpid} in JSON."

Tool record (may contain all needed fields):
{tool_json}

Task:
1) Use the tool record as the source of information to answer the user's request.
2) Return a JSON object with EXACTLY these keys:
- material_id (string)
- band_gap_eV (number)
- formation_energy_per_atom_eV (number)
- is_metal (boolean)
- energy_above_hull_eV (number)
- is_stable (boolean)

Constraints that must hold:
- If is_metal is true, band_gap_eV must be <= 0.1
- If is_stable is true, energy_above_hull_eV must be <= 1e-6

Output ONLY the JSON. No extra text.
"""

def main():
    df = pd.read_csv(CSV_PATH)

    need = ["material_id","band_gap","formation_energy_per_atom","is_metal","energy_above_hull","is_stable"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"gt_200_full missing cols={missing}, got={df.columns.tolist()}")

    rows = []
    t0 = time.time()

    for i, r in enumerate(df.to_dict(orient="records"), 1):
        mpid = str(r["material_id"])
        tool_record = build_tool_record(r)
        prompt = build_prompt(mpid, tool_record)

        try:
            resp = model.generate_content(prompt)
            raw = (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            raw = f"Error: {e}"

        rows.append({
            "mp_id": mpid,
            "tool_record_json": json.dumps(tool_record, ensure_ascii=False),
            "raw_output": raw,
            "model": "gemini",
            "experiment": EXPERIMENT,
            "ts": time.time(),
        })

        if i % 50 == 0:
            print(f"[{i}/{len(df)}] last={mpid}")

        time.sleep(SLEEP_SEC)

    out_path = OUT_DIR / "gemini_exp2_hard_grounded_novfy.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)
    print("Total seconds:", round(time.time() - t0, 2))

if __name__ == "__main__":
    main()