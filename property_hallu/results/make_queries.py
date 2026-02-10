# experiments/property_hallu/make_queries.py
import json
from mp_api.client import MPRester
import os

key = os.getenv("PMG_MAPI_KEY") or os.getenv("MP_API_KEY")
assert key, "Missing MP API key in env"

# 你先固定这三个，最稳：数值+布尔混合
fields = ["band_gap", "formation_energy_per_atom", "is_metal"]

# 抽 50 个材料（尽量简单：先从 summary 里抓，不加太多筛选）
N_ID = 50

with MPRester(key) as mpr:
    docs = mpr.materials.summary.search(fields=["material_id"], limit=N_ID)
    material_ids = [d.material_id for d in docs]

out = []
qid = 0
for mid in material_ids:
    for f in fields:
        out.append({
            "query_id": f"q{qid:04d}",
            "material_id": mid,
            "field": f,
            "question": f'Use Materials Project only. What is the value of "{f}" for material {mid}? Return one short sentence with the value.'
        })
        qid += 1

path = "experiments/property_hallu/queries.jsonl"
with open(path, "w") as w:
    for r in out:
        w.write(json.dumps(r) + "\n")

print("Wrote", len(out), "queries to", path)
print("Example:", out[0])