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

MODEL_NAME  = "gemini-2.5-flash-lite"
EXPERIMENT  = "exp1_hard_baseline"
SLEEP_SEC   = 0.6

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY)")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

REQUIRED_KEYS = [
    "material_id",
    "band_gap_eV",
    "formation_energy_per_atom_eV",
    "is_metal",
    "energy_above_hull_eV",
    "is_stable",
]

def build_prompt(mpid: str) -> str:
    return f"""You are a materials assistant.

User asks: "Return the following properties for {mpid} in JSON."

Return a JSON object with EXACTLY these keys:
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
    if "material_id" not in df.columns:
        raise ValueError(f"gt_200_full missing material_id. cols={df.columns.tolist()}")

    ids = df["material_id"].astype(str).tolist()
    print(f"Loaded {len(ids)} materials")

    rows = []
    t0 = time.time()
    for i, mpid in enumerate(ids, 1):
        prompt = build_prompt(mpid)
        try:
            resp = model.generate_content(prompt)
            raw = (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            raw = f"Error: {e}"

        rows.append({
            "mp_id": mpid,
            "raw_output": raw,
            "model": "gemini",
            "experiment": EXPERIMENT,
            "ts": time.time(),
        })

        if i % 50 == 0:
            print(f"[{i}/{len(ids)}] last={mpid}")

        time.sleep(SLEEP_SEC)

    out_path = OUT_DIR / "gemini_exp1_hard_baseline.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)
    print("Total seconds:", round(time.time() - t0, 2))

if __name__ == "__main__":
    main()