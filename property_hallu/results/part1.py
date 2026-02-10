# %%
from dotenv import load_dotenv
import os, pathlib

ENV_PATH = str(pathlib.Path.home() / "llamp" / ".env.local")
load_dotenv(ENV_PATH, override=True)

for k in ["PMG_MAPI_KEY","MP_API_KEY","GOOGLE_API_KEY"]:
    v = os.getenv(k)
    print(k, "SET" if v else "MISSING", f"(len={len(v)})" if v else "")

# %%
import os, json, time, re, random
from pathlib import Path
import numpy as np
import pandas as pd

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
random.seed(42)
np.random.seed(42)

ROOT = Path(".")
DATA_DIR = ROOT / "data"
PROMPT_DIR = ROOT / "prompts"
LOG_DIR = ROOT / "logs"
RES_DIR = ROOT / "results"
ANN_DIR = ROOT / "annotation"

for d in [DATA_DIR, PROMPT_DIR, LOG_DIR, RES_DIR, ANN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("RUN_ID =", RUN_ID)

# %%
from mp_api.client import MPRester

# 你之前用过：PMG_MAPI_KEY / MP_API_KEY
MP_KEY = os.getenv("PMG_MAPI_KEY") or os.getenv("MP_API_KEY")
assert MP_KEY, "Missing MP API key env: set PMG_MAPI_KEY or MP_API_KEY"

from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_MODEL = "models/gemini-2.5-flash-lite"  # 你 list_models 里有这个
top_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.0,
    convert_system_message_to_human=True,  # 避免 SystemMessage 报错
)
print("top_llm built:", GEMINI_MODEL)

# %%
def write_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def parse_first_float(text: str):
    m = re.search(FLOAT_RE, text)
    return float(m.group(0)) if m else None

def parse_mpid(text: str):
    # very simple mpid extractor
    m = re.search(r"\bmp-\d+\b", text)
    return m.group(0) if m else None

def is_refusal(text: str):
    t = (text or "").lower()
    keywords = ["i can't", "cannot", "can't", "unable", "don't know", "not sure", "no data", "not available"]
    return any(k in t for k in keywords)

# %%
import yaml

TASKS = {
    "band_gap": {
        "field": "band_gap",
        "unit": "eV",
        "tol": 0.05,
        "templates": [
            "What is the band gap (eV) of material {mpid}? Answer with a single number.",
            "Give the band gap of {mpid} in eV. No extra text.",
            "Provide the band gap for {mpid} (unit: eV). Only the number."
        ],
    },
    "formation_energy_per_atom": {
        "field": "formation_energy_per_atom",
        "unit": "eV/atom",
        "tol": 0.05,
        "templates": [
            "What is the formation energy per atom (eV/atom) of material {mpid}? Answer with a single number.",
            "Give the formation energy per atom of {mpid} in eV/atom. No extra text.",
            "Provide formation_energy_per_atom for {mpid} (unit: eV/atom). Only the number."
        ],
    },
}

tmpl_path = PROMPT_DIR / "templates.yaml"
yaml.safe_dump({k: v["templates"] for k, v in TASKS.items()}, tmpl_path.open("w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
print("Saved templates:", tmpl_path)

# %%
from mp_api.client import MPRester
import os, pandas as pd

key = os.getenv("PMG_MAPI_KEY") or os.getenv("MP_API_KEY")
gt = pd.read_csv("gt_200.csv")  # 你刚保存的
ids = gt["material_id"].astype(str).tolist()

FIELDS_EXTRA = ["material_id","is_metal","energy_above_hull","is_stable"]

rows = []
with MPRester(key) as mpr:
    docs = mpr.materials.summary.search(material_ids=ids, fields=FIELDS_EXTRA)
    for d in docs:
        rows.append({
            "material_id": str(d.material_id),
            "is_metal": d.is_metal,
            "energy_above_hull": d.energy_above_hull,
            "is_stable": d.is_stable
        })

extra = pd.DataFrame(rows)
gt_full = gt.merge(extra, on="material_id", how="left")
gt_full.to_csv("gt_200_full.csv", index=False)
print("Saved: gt_200_full.csv", gt_full.columns.tolist())

# %%
import pandas as pd
from pathlib import Path

# 改成你的真实路径
GT_BG_PATH = Path("/Users/yutongduan/llamp/experiments/property_hallu/results/gt_200_full.csv")
GT_FE_PATH = Path("/Users/yutongduan/llamp/experiments/property_hallu/results/gt_200_full.csv")  

assert GT_BG_PATH.exists(), f"not found: {GT_BG_PATH}"
assert GT_FE_PATH.exists(), f"not found: {GT_FE_PATH}"

gt_bg = pd.read_csv(GT_BG_PATH)
gt_fe = pd.read_csv(GT_FE_PATH)

# 统一列名（有些文件叫 gt_fe_per_atom / formation_energy_per_atom）
rename_fe = {}
if "formation_energy_per_atom" not in gt_fe.columns and "gt_fe_per_atom" in gt_fe.columns:
    rename_fe["gt_fe_per_atom"] = "formation_energy_per_atom"
gt_fe = gt_fe.rename(columns=rename_fe)

# 只保留我们需要的列
gt_bg_keep = ["material_id","band_gap","is_metal","energy_above_hull","is_stable"]
gt_fe_keep = ["material_id","formation_energy_per_atom"]

missing_bg = [c for c in gt_bg_keep if c not in gt_bg.columns]
missing_fe = [c for c in gt_fe_keep if c not in gt_fe.columns]
assert not missing_bg, f"gt_bg missing: {missing_bg}, cols={gt_bg.columns.tolist()}"
assert not missing_fe, f"gt_fe missing: {missing_fe}, cols={gt_fe.columns.tolist()}"

gt_bg = gt_bg[gt_bg_keep].copy()
gt_fe = gt_fe[gt_fe_keep].copy()

# merge：以 bandgap 那个为主（你的样本集通常是从它来的）
gt = gt_bg.merge(gt_fe, on="material_id", how="left")

print("gt_bg:", len(gt_bg), "gt_fe:", len(gt_fe), "merged gt:", len(gt))
print("missing formation_energy_per_atom after merge:", gt["formation_energy_per_atom"].isna().sum())

gt.head()

# %%
FIELDS_GT = ["material_id", "band_gap", "formation_energy_per_atom", "is_metal", "energy_above_hull", "is_stable"]

GT_PATH = DATA_DIR / "mp_ground_truth_seed42.csv"
IDS_PATH = DATA_DIR / "mp_sample_ids_seed42.json"

def fetch_gt_for_ids(material_ids):
    rows = []
    with MPRester(MP_KEY) as mpr:
        # batch=100 通常没问题
        for i in range(0, len(material_ids), 100):
            batch = material_ids[i:i+100]
            docs = mpr.materials.summary.search(material_ids=batch, fields=FIELDS_GT)
            for d in docs:
                rows.append({
                    "material_id": str(d.material_id),
                    "band_gap": d.band_gap,
                    "formation_energy_per_atom": d.formation_energy_per_atom,
                    "is_metal": d.is_metal,
                    "energy_above_hull": d.energy_above_hull,
                    "is_stable": d.is_stable,
                })
    gt = pd.DataFrame(rows).dropna(subset=["material_id"]).drop_duplicates("material_id")
    return gt



gt.head()

# %%
import pandas as pd
from pathlib import Path

BG_PATH = Path("/Users/yutongduan/llamp/experiments/property_hallu/results/baseline_band_gap_200.csv")
FE_PATH = Path("/Users/yutongduan/llamp/experiments/property_hallu/results/baseline_fe_200.csv")

bg = pd.read_csv(BG_PATH)
fe = pd.read_csv(FE_PATH)

print("bg cols:", bg.columns.tolist())
print("fe cols:", fe.columns.tolist())

# 统一列名
bg = bg.rename(columns={"gt_band_gap": "band_gap"})
fe = fe.rename(columns={
    "gt_fe_per_atom": "formation_energy_per_atom",
    "gt_formation_energy_per_atom": "formation_energy_per_atom"
})

need_bg = ["material_id", "band_gap"]
need_fe = ["material_id", "formation_energy_per_atom"]

assert all(c in bg.columns for c in need_bg), f"bg missing {set(need_bg)-set(bg.columns)}"
assert all(c in fe.columns for c in need_fe), f"fe missing {set(need_fe)-set(fe.columns)}"

gt_200 = (
    bg[need_bg]
    .merge(fe[need_fe], on="material_id", how="inner")
    .drop_duplicates("material_id")
    .reset_index(drop=True)
)

print("gt_200 size:", len(gt_200))
gt_200.to_csv("gt_200.csv", index=False)
print("Saved: gt_200.csv")
gt_200.head()

# %%
import pandas as pd

# 直接用你已经存下来的固定样本集
gt = pd.read_csv("gt_200_full.csv")

# 这就是你的 ids list（200 个）
ids = gt["material_id"].astype(str).dropna().unique().tolist()
print("Use GT ids:", len(ids))

# 如果后面代码还需要 N
N = len(ids)

FIELDS_GT = ["material_id", "band_gap", "formation_energy_per_atom", "is_metal", "energy_above_hull", "is_stable"]

GT_PATH = DATA_DIR / "mp_ground_truth_seed42.csv"
IDS_PATH = DATA_DIR / "mp_sample_ids_seed42.json"

def fetch_gt_for_ids(material_ids):
    rows = []
    with MPRester(MP_KEY) as mpr:
        # batch=100 通常没问题
        for i in range(0, len(material_ids), 100):
            batch = material_ids[i:i+100]
            docs = mpr.materials.summary.search(material_ids=batch, fields=FIELDS_GT)
            for d in docs:
                rows.append({
                    "material_id": str(d.material_id),
                    "band_gap": d.band_gap,
                    "formation_energy_per_atom": d.formation_energy_per_atom,
                    "is_metal": d.is_metal,
                    "energy_above_hull": d.energy_above_hull,
                    "is_stable": d.is_stable,
                })
    gt = pd.DataFrame(rows).dropna(subset=["material_id"]).drop_duplicates("material_id")
    return gt

gt.head()

# %%
def run_baseline(task_name: str, gt_df: pd.DataFrame, out_jsonl: Path):
    cfg = TASKS[task_name]
    tol = cfg["tol"]
    templates = cfg["templates"]
    field = cfg["field"]

    for _, r in gt_df.iterrows():
        mpid = str(r["material_id"])
        gt_val = r[field]

        tid = random.randrange(len(templates))
        query = templates[tid].format(mpid=mpid)

        t0 = time.time()
        resp = top_llm.invoke(query)
        raw = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
        latency = time.time() - t0

        pred = parse_first_float(raw)
        refusal = is_refusal(raw)
        parse_fail = (pred is None)

        # 评价
        if parse_fail:
            label = "parse_fail"
        elif refusal:
            label = "refusal"
        else:
            abs_err = abs(float(pred) - float(gt_val))
            label = "ok" if abs_err <= tol else "wrong_value"

        rec = {
            "run_id": RUN_ID,
            "setting": "A_no_tool",
            "task": task_name,
            "material_id": mpid,
            "template_id": tid,
            "query": query,
            "gt_value": None if pd.isna(gt_val) else float(gt_val),
            "raw_output": raw,
            "pred_value": pred,
            "refusal": bool(refusal),
            "parse_fail": bool(parse_fail),
            "label": label,
            "latency_s": latency,
            "ts": time.time(),
        }
        write_jsonl(out_jsonl, rec)

    print("baseline done ->", out_jsonl)

baseline_jsonl_bg = LOG_DIR / f"{RUN_ID}_baseline_band_gap.jsonl"
baseline_jsonl_fe = LOG_DIR / f"{RUN_ID}_baseline_fe.jsonl"

# 你今天要“直接跑完”，就两条都跑：
run_baseline("band_gap", gt, baseline_jsonl_bg)
run_baseline("formation_energy_per_atom", gt, baseline_jsonl_fe)

# %%
def run_grounded(task_name: str, gt_df: pd.DataFrame, out_jsonl: Path):
    cfg = TASKS[task_name]
    field = cfg["field"]
    unit = cfg["unit"]
    tol = cfg["tol"]

    # 先批量取 tool 值（就是 MP summary 里对应字段）
    tool_rows = []
    ids = gt_df["material_id"].astype(str).tolist()

    with MPRester(MP_KEY) as mpr:
        for i in range(0, len(ids), 100):
            batch = ids[i:i+100]
            docs = mpr.materials.summary.search(material_ids=batch, fields=["material_id", field])
            for d in docs:
                tool_rows.append({"material_id": str(d.material_id), "tool_value": getattr(d, field)})

    tool_df = pd.DataFrame(tool_rows).dropna(subset=["tool_value"]).drop_duplicates("material_id")
    run = gt_df.merge(tool_df, on="material_id", how="inner")
    assert len(run) == len(gt_df), "Some mpids missing tool value"

    for _, r in run.iterrows():
        mpid = str(r["material_id"])
        gt_val = float(r[field])
        tool_val = float(r["tool_value"])

        prompt = f"""You are given a Materials Project value.
RULES:
- Use ONLY the given number.
- Output one sentence with the value in {unit}.
Given: {field} = {tool_val}
Material: {mpid}
"""

        t0 = time.time()
        resp = top_llm.invoke(prompt)
        raw = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
        latency = time.time() - t0

        pred = parse_first_float(raw)
        parse_fail = (pred is None)
        mismatch_vs_tool = (parse_fail or (abs(float(pred) - tool_val) > 1e-12))

        abs_err = None if parse_fail else abs(float(pred) - gt_val)
        acc_ok = (abs_err is not None) and (abs_err <= tol)

        rec = {
            "run_id": RUN_ID,
            "setting": "B_tool_only",
            "task": task_name,
            "material_id": mpid,
            "gt_value": gt_val,
            "tool_value": tool_val,
            "raw_output": raw,
            "pred_value": pred,
            "parse_fail": bool(parse_fail),
            "mismatch_vs_tool": bool(mismatch_vs_tool),
            "label": "ok" if acc_ok else ("parse_fail" if parse_fail else "mismatch_or_wrong"),
            "latency_s": latency,
            "ts": time.time(),
        }
        write_jsonl(out_jsonl, rec)

    print("grounded done ->", out_jsonl)

grounded_jsonl_bg = LOG_DIR / f"{RUN_ID}_grounded_band_gap.jsonl"
grounded_jsonl_fe = LOG_DIR / f"{RUN_ID}_grounded_fe.jsonl"

run_grounded("band_gap", gt, grounded_jsonl_bg)
run_grounded("formation_energy_per_atom", gt, grounded_jsonl_fe)

# %%
def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def prelabel_error_type(df: pd.DataFrame) -> pd.DataFrame:
    # 只对 baseline 错误做预标注
    df = df.copy()
    df["error_type"] = ""
    df["notes"] = ""

    # 基础类
    df.loc[df["parse_fail"] == True, "error_type"] = "format_or_parse_fail"
    df.loc[(df["parse_fail"] == False) & (df["refusal"] == True), "error_type"] = "refusal"

    # 数值错：进一步分 “mode_guess” vs “other_wrong”
    wrong_mask = (df["label"] == "wrong_value")
    if wrong_mask.any():
        # 找众数（四舍五入到 2 位避免浮点碎裂）
        tmp = df.loc[wrong_mask & df["pred_value"].notna(), "pred_value"].round(2)
        if len(tmp) > 0:
            top_modes = tmp.value_counts().head(5).index.tolist()
        else:
            top_modes = []

        def is_mode_guess(v):
            if pd.isna(v): 
                return False
            return round(float(v), 2) in set(top_modes)

        df.loc[wrong_mask & df["pred_value"].apply(is_mode_guess), "error_type"] = "mode_guess"
        df.loc[wrong_mask & (df["error_type"] == ""), "error_type"] = "wrong_value_other"

    # （可选）单位错误：这里用启发式（输出里包含 unit 但错/或不含 unit），你也可以更严格
    # 先不强行判 unit_error，留给人工复核更稳

    return df

# baseline 两个任务合并做标注池
df_bg = load_jsonl(baseline_jsonl_bg)
df_fe = load_jsonl(baseline_jsonl_fe)
df_base = pd.concat([df_bg, df_fe], ignore_index=True)

df_base_labeled = prelabel_error_type(df_base)
print(df_base_labeled["task"].value_counts())
print(df_base_labeled["error_type"].value_counts().head(10))

to_annotate = df_base_labeled[df_base_labeled["label"] != "ok"].copy()
out_ann = ANN_DIR / f"{RUN_ID}_to_annotate.csv"
to_annotate.to_csv(out_ann, index=False)
print("Export to annotate:", out_ann)

# %%
from sklearn.metrics import cohen_kappa_score

# 你可以把同一个 to_annotate.csv 复制两份给两个人标：
# annotator1.csv, annotator2.csv （要求同一行 material_id/task 对齐）
ann1_path = ANN_DIR / "annotator1.csv"
ann2_path = ANN_DIR / "annotator2.csv"

if ann1_path.exists() and ann2_path.exists():
    a1 = pd.read_csv(ann1_path)
    a2 = pd.read_csv(ann2_path)
    key_cols = ["run_id","task","material_id","query"]
    a1 = a1.sort_values(key_cols).reset_index(drop=True)
    a2 = a2.sort_values(key_cols).reset_index(drop=True)

    kappa = cohen_kappa_score(a1["error_type"].astype(str), a2["error_type"].astype(str))
    print("Cohen's kappa =", kappa)
else:
    print("Skip kappa: annotator1/2 files not found.")

# %%
# 从 gt 里挑 10 个 band_gap 在[1,3] 且稳定的材料
pool = gt.copy()
pool = pool[(pool["band_gap"].notna()) & (pool["band_gap"] >= 1.0) & (pool["band_gap"] <= 3.0)]
if "is_stable" in pool.columns:
    pool = pool[pool["is_stable"] == True]

pool = pool.sample(n=min(10, len(pool)), random_state=42).reset_index(drop=True)
pool[["material_id","band_gap","formation_energy_per_atom"]]

# %%
def high_order_query(pool_df: pd.DataFrame, setting: str):
    """
    setting:
      - "A_no_tool": 只问，不给任何结构化工具信息
      - "B_list_grounded": 给候选列表(含 band_gap)，要求从中选择，再给 formation energy（模型仍可能编）
      - "C_verify_retry": 先 A 回答，再给它正确的 tool 值让它改一次（模拟 self-correcting）
    """
    # 准备候选表（只给 band_gap，逼它先做 step1）
    rows_txt = "\n".join([f"- {r.material_id}: band_gap={r.band_gap:.4f} eV" for r in pool_df.itertuples()])
    q = f"""You must choose ONE material_id from the candidate list whose band gap is between 1 and 3 eV (inclusive).
Then report its formation energy per atom in eV/atom.

Candidate list:
{rows_txt}

Output format (two lines):
material_id: <mp-...>
formation_energy_per_atom: <number>
"""

    if setting == "A_no_tool":
        prompt = "Find a stable material with band gap between 1 and 3 eV, then report its formation energy per atom (eV/atom)."
    elif setting == "B_list_grounded":
        prompt = q
    else:
        raise ValueError("unknown setting")

    t0 = time.time()
    resp = top_llm.invoke(prompt)
    raw = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
    latency = time.time() - t0

    chosen = parse_mpid(raw)
    pred_fe = parse_first_float(raw)

    # 评分：step1 chosen 是否在 pool 且满足 bandgap
    ok_choose = False
    gt_fe = None
    if chosen is not None:
        hit = pool_df[pool_df["material_id"].astype(str) == chosen]
        if len(hit) == 1:
            bg = float(hit.iloc[0]["band_gap"])
            ok_choose = (1.0 <= bg <= 3.0)
            gt_fe = float(hit.iloc[0]["formation_energy_per_atom"])

    # step2 formation energy 是否正确（容忍 0.05 eV/atom）
    tol = 0.05
    ok_fe = (pred_fe is not None) and (gt_fe is not None) and (abs(float(pred_fe) - float(gt_fe)) <= tol)

    return {
        "setting": setting,
        "raw_output": raw,
        "chosen_mpid": chosen,
        "pred_fe": pred_fe,
        "ok_choose": ok_choose,
        "gt_fe": gt_fe,
        "ok_fe": ok_fe,
        "latency_s": latency,
    }

# 跑 30 次（同一个 pool，每次看模型稳定性/崩坏模式）
H = []
for _ in range(30):
    H.append(high_order_query(pool, "A_no_tool"))
    H.append(high_order_query(pool, "B_list_grounded"))

dfH = pd.DataFrame(H)
print(dfH.groupby("setting")[["ok_choose","ok_fe"]].mean())

outH = RES_DIR / f"{RUN_ID}_high_order_H1.csv"
dfH.to_csv(outH, index=False)
print("saved:", outH)

# %%
def high_order_verify_retry(pool_df: pd.DataFrame):
    # 先 baseline 问
    first = high_order_query(pool_df, "B_list_grounded")  # 用候选列表更可判
    chosen = first["chosen_mpid"]

    # 如果选错/没选出来，直接记为失败
    if not first["ok_choose"] or chosen is None:
        return {**first, "setting": "C_verify_retry", "retry_ok_fe": False, "retry_pred_fe": None, "retry_raw": ""}

    # 给它 chosen 的“正确 formation energy”让它修正一次
    hit = pool_df[pool_df["material_id"].astype(str) == chosen].iloc[0]
    tool_fe = float(hit["formation_energy_per_atom"])

    prompt = f"""You previously answered:
{first["raw_output"]}

Now you are given the correct tool value:
formation_energy_per_atom({chosen}) = {tool_fe} eV/atom

Please output again (two lines):
material_id: {chosen}
formation_energy_per_atom: <number>
Use ONLY the tool value. Do not guess.
"""
    t0 = time.time()
    resp = top_llm.invoke(prompt)
    retry_raw = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
    retry_pred = parse_first_float(retry_raw)

    tol = 0.05
    retry_ok = (retry_pred is not None) and (abs(float(retry_pred) - tool_fe) <= tol)

    return {
        **first,
        "setting": "C_verify_retry",
        "tool_fe": tool_fe,
        "retry_raw": retry_raw,
        "retry_pred_fe": retry_pred,
        "retry_ok_fe": retry_ok,
        "retry_latency_s": time.time() - t0
    }

H2 = [high_order_verify_retry(pool) for _ in range(30)]
dfH2 = pd.DataFrame(H2)
print("choose ok rate =", dfH2["ok_choose"].mean())
print("retry ok_fe rate =", dfH2["retry_ok_fe"].mean())

outH2 = RES_DIR / f"{RUN_ID}_high_order_self_correct.csv"
dfH2.to_csv(outH2, index=False)
print("saved:", outH2)

# %%
#抽样120条wrong_value_other

import pandas as pd

df = pd.read_csv("annotation/20251224_103747_to_annotate.csv")

# 只抽 wrong_value_other 来精标
sub = df[df["error_type"]=="wrong_value_other"].sample(n=min(120, (df["error_type"]=="wrong_value_other").sum()), random_state=42)

sub.to_csv("annotation/to_refine_120.csv", index=False)
print("saved annotation/to_refine_120.csv", "size=", len(sub))

# %%
from mp_api.client import MPRester
import os
key = os.getenv("PMG_MAPI_KEY") or os.getenv("MP_API_KEY")

with MPRester(key) as mpr:
    d = mpr.materials.summary.search(material_ids=["mp-19"], fields=["formula_pretty","formation_energy_per_atom"])[0]
    print(d.formula_pretty, d.formation_energy_per_atom)

# %%
import pandas as pd
from pathlib import Path

# 路径按你现在的文件位置改一下即可
FULL = "annotation/20251224_103747_to_annotate.csv"          # 全量（含自动 error_type）
REF  = "annotation/refined_annotation.csv"          # 你刚导出的 CSV（含你在 Q 列的精标）

full = pd.read_csv(FULL)
ref  = pd.read_csv(REF)

# === 1) 统一列名（兼容你文件里可能叫 Q / error_type_refined）===
# 你说“标注在 Q 列”，这里把它映射成 error_type_refined
if "error_type_refined" not in ref.columns:
    if "Q" in ref.columns:
        ref = ref.rename(columns={"Q": "error_type_refined"})
    elif "error_type_final" in ref.columns:
        ref = ref.rename(columns={"error_type_final": "error_type_refined"})

# notes 也做个兼容
if "notes" not in ref.columns:
    if "Notes" in ref.columns:
        ref = ref.rename(columns={"Notes": "notes"})
    else:
        ref["notes"] = ""

# === 2) 用这些 key 做 join（跟你之前一致）===
key = ["run_id","setting","task","material_id","template_id"]

missing_key = [c for c in key if c not in full.columns]
assert not missing_key, f"FULL missing columns: {missing_key}"

missing_key_ref = [c for c in key if c not in ref.columns]
assert not missing_key_ref, f"REF missing columns: {missing_key_ref}. REF cols={ref.columns.tolist()}"

need_ref_cols = key + ["error_type_refined","notes"]
missing_ref_cols = [c for c in need_ref_cols if c not in ref.columns]
assert not missing_ref_cols, f"REF missing cols: {missing_ref_cols}. REF cols={ref.columns.tolist()}"

ref = ref[need_ref_cols].copy()

# === 3) merge：有精标就覆盖自动标签，否则用自动 error_type ===
m = full.merge(ref, on=key, how="left")

# 处理空白/NaN
for col in ["error_type_refined","error_type"]:
    if col in m.columns:
        m[col] = m[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

m["error_type_final"] = m["error_type_refined"].fillna(m["error_type"])

# === 4) 只统计 baseline(A_no_tool) 且 label != ok 的错误 ===
assert "setting" in m.columns and "label" in m.columns, f"Need columns setting/label. cols={m.columns.tolist()}"

err = m[m["setting"].astype(str).str.contains("no_tool", na=False) & (m["label"] != "ok")].copy()

dist = (
    err.groupby(["task","error_type_final"])
       .size()
       .reset_index(name="n")
)
dist["pct_in_task"] = dist["n"] / dist.groupby("task")["n"].transform("sum")
dist = dist.sort_values(["task","pct_in_task"], ascending=[True, False])

out_path = Path("results/error_type_distribution.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
dist.to_csv(out_path, index=False)

print("Saved:", out_path.resolve())
print(dist)

# %%



