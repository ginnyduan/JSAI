import pandas as pd
import json

PATH = "/Users/yutongduan/llamp/experiments/property_hallu/hard_task/results/results_hard_v1/gemini/exp3/metrics_gemini_exp3_hard_verify_override.csv"
df = pd.read_csv(PATH)

print("n =", len(df))
print("acc =", df["is_correct"].mean())

# 错误分解
err = df[df["is_correct"] == False]
print("\nError counts:")
for c in ["refusal","parse_fail","schema_fail","value_mismatch","constraint_violation"]:
    print(c, err[c].mean(), err[c].sum())

# 看看错误样本的 mp_id
print("\nFirst 20 wrong mp_ids:")
print(err["mp_id"].head(20).tolist())