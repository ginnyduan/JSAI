import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


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


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive match
    lower = {x.lower(): x for x in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def compute_refusal(raw: str) -> bool:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return False
    s = str(raw)
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, s, flags=re.I):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="master_experiment_results_v4.csv")
    ap.add_argument("--out", default="results_v4", help="output dir")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.master)

    # --- column mapping (robust) ---
    col_model = find_col(df, ["model", "llm", "model_name"])
    col_exp = find_col(df, ["exp", "experiment", "experiment_id", "method", "setting"])
    col_task = find_col(df, ["task", "property"])
    col_correct = find_col(df, ["is_correct", "correct", "match", "ok"])
    col_pred = find_col(df, ["pred_value", "pred", "prediction", "parsed_value", "pred_band_gap", "pred_fe_per_atom"])
    col_raw = find_col(df, ["raw_output", "raw_text", "output", "response"])
    col_parse_fail = find_col(df, ["parse_fail", "parse_error", "format_fail"])
    col_refusal = find_col(df, ["refusal", "is_refusal"])

    # sanity
    need = [col_model, col_exp, col_task]
    if any(x is None for x in need):
        raise ValueError(
            f"Missing key columns. Found: model={col_model}, exp={col_exp}, task={col_task}. "
            f"All cols={df.columns.tolist()}"
        )

    # normalize booleans
    if col_correct is None:
        # infer from abs_err/tolerance if exists? fallback: always False
        print("[WARN] is_correct not found; will infer from (abs_err <= tol) if abs_err exists, else False.")
        col_abs = find_col(df, ["abs_err", "absolute_error", "err"])
        if col_abs is not None:
            # default tol: 0.05 (you can later overwrite per task)
            df["_is_correct"] = df[col_abs].astype(float) <= 0.05
            col_correct = "_is_correct"
        else:
            df["_is_correct"] = False
            col_correct = "_is_correct"

    df[col_correct] = df[col_correct].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(df[col_correct])

    if col_parse_fail is None:
        # infer parse_fail from pred_value
        if col_pred is None:
            print("[WARN] pred_value not found; parse_fail will be inferred from raw_output numeric existence.")
            # if raw exists but no pred, treat as parse_fail=True
            df["_parse_fail"] = df[col_raw].isna() if col_raw else True
        else:
            df["_parse_fail"] = pd.to_numeric(df[col_pred], errors="coerce").isna()
        col_parse_fail = "_parse_fail"
    else:
        df[col_parse_fail] = df[col_parse_fail].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(df[col_parse_fail])

    if col_refusal is None:
        if col_raw is None:
            df["_refusal"] = False
        else:
            df["_refusal"] = df[col_raw].apply(compute_refusal)
        col_refusal = "_refusal"
    else:
        df[col_refusal] = df[col_refusal].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(df[col_refusal])

    # Answered = (not parse_fail) AND (not refusal)
    df["_answered"] = (~df[col_parse_fail].astype(bool)) & (~df[col_refusal].astype(bool))

    # --- metrics ---
    group_cols = [col_model, col_exp, col_task]
    g = df.groupby(group_cols, dropna=False)

    res = g.agg(
        n=("__dummy__", "size") if "__dummy__" in df.columns else (col_task, "size"),
        answer_rate=("_answered", "mean"),
        parse_fail_rate=(col_parse_fail, "mean"),
        refusal_rate=(col_refusal, "mean"),
        overall_acc=(col_correct, "mean"),
    ).reset_index()

    # conditional accuracy (given answered)
    def cond_acc(sub):
        answered = sub["_answered"].astype(bool)
        if answered.sum() == 0:
            return np.nan
        return sub.loc[answered, col_correct].astype(bool).mean()

    res["acc_given_answered"] = g.apply(cond_acc).values

    # rename to standard headers for paper
    res = res.rename(columns={
        col_model: "model",
        col_exp: "exp",
        col_task: "task",
    })

    res = res.sort_values(["task", "model", "exp"]).reset_index(drop=True)

    out_path = outdir / "metrics_by_model_exp_task.csv"
    res.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(res.head(20).to_string(index=False))


if __name__ == "__main__":
    main()