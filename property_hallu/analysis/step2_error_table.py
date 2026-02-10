import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {x.lower(): x for x in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def normalize_error_type(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    if s == "":
        return ""
    # unify names (edit these mappings to match your labels)
    s = s.replace("wrong_value_other", "wrong_value_other")
    s = s.replace("mode guess", "mode_guess").replace("mode_guess", "mode_guess")
    s = s.replace("sign error", "sign_error").replace("sign_error", "sign_error")
    s = s.replace("scale error", "scale_error").replace("scale_error", "scale_error")
    s = s.replace("schema mismatch", "schema_mismatch")
    s = s.replace("format", "format_error")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="master_experiment_results_v4.csv")
    ap.add_argument("--out", default="results_v4", help="output dir")
    ap.add_argument("--focus_setting", default=None,
                    help="Optional filter: only include rows where setting/exp contains this string (e.g., 'baseline' or 'no_tool').")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.master)

    col_model = find_col(df, ["model", "llm", "model_name"])
    col_exp = find_col(df, ["exp", "experiment", "experiment_id", "method", "setting"])
    col_task = find_col(df, ["task", "property"])
    col_label = find_col(df, ["label", "result_label"])
    col_error = find_col(df, ["error_type_final", "error_type", "error", "error_label"])
    col_correct = find_col(df, ["is_correct", "correct", "match", "ok"])
    col_parse_fail = find_col(df, ["parse_fail", "parse_error", "format_fail"])
    col_refusal = find_col(df, ["refusal", "is_refusal"])

    # If error_type_final not present, fall back to error_type
    if col_error is None:
        raise ValueError(f"No error_type column found. cols={df.columns.tolist()}")

    # Normalize
    df["_error_type"] = df[col_error].apply(normalize_error_type)

    # optional focus filter
    if args.focus_setting and col_exp:
        df = df[df[col_exp].astype(str).str.contains(args.focus_setting, case=False, na=False)].copy()

    # define "error" rows: not correct OR parse_fail OR refusal
    if col_correct is not None:
        corr = df[col_correct].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        corr = corr.fillna(df[col_correct])
        df["_is_correct"] = corr.astype(bool)
    else:
        df["_is_correct"] = False

    if col_parse_fail is not None:
        pf = df[col_parse_fail].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        df["_parse_fail"] = pf.fillna(df[col_parse_fail]).astype(bool)
    else:
        df["_parse_fail"] = False

    if col_refusal is not None:
        rf = df[col_refusal].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        df["_refusal"] = rf.fillna(df[col_refusal]).astype(bool)
    else:
        df["_refusal"] = False

    df["_is_error"] = (~df["_is_correct"]) | df["_parse_fail"] | df["_refusal"]

    # fill empty error_type for error rows
    df.loc[(df["_is_error"]) & (df["_error_type"] == ""), "_error_type"] = "unclassified"

    # distribution
    group_cols = [col_task, col_model, col_exp, "_error_type"]
    dist = (
        df[df["_is_error"]]
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n")
    )

    # pct within each (task, model, exp)
    dist["pct_in_group"] = dist["n"] / dist.groupby([col_task, col_model, col_exp])["n"].transform("sum")

    dist = dist.rename(columns={
        col_task: "task",
        col_model: "model",
        col_exp: "exp",
        "_error_type": "error_type",
    }).sort_values(["task", "model", "exp", "pct_in_group"], ascending=[True, True, True, False])

    out1 = outdir / "error_type_distribution.csv"
    dist.to_csv(out1, index=False)
    print("Saved:", out1)

    # pivot for "big table" (easier to paste into paper)
    pivot = dist.pivot_table(
        index=["task", "error_type"],
        columns=["model", "exp"],
        values="pct_in_group",
        aggfunc="first",
        fill_value=0.0,
    ).reset_index()

    out2 = outdir / "error_type_pivot.csv"
    pivot.to_csv(out2, index=False)
    print("Saved:", out2)

    # optional: export a small sample for manual refinement
    # (top 200 error rows)
    sample_cols = ["task", "model", "exp"]
    keep_cols = [c for c in [col_task, col_model, col_exp, col_label, col_error] if c is not None]
    extra = [c for c in ["material_id", "query", "gt_value", "pred_value", "raw_output"] if c in df.columns]
    sample_df = df[df["_is_error"]].copy()
    sample_df = sample_df[keep_cols + extra].head(200)
    out3 = outdir / "samples_for_labeling.csv"
    sample_df.to_csv(out3, index=False)
    print("Saved:", out3)


if __name__ == "__main__":
    main()