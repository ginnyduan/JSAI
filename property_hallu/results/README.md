# LLaMP 幻觉研究实验（Property Retrieval + 少量 High-order 加分实验）

这份文件夹包含我为 **Materials Project (MP)** 属性检索任务做的“幻觉分析 + 修复”实验产物（适合 4 页 short paper），并额外做了少量 **high-order** / **self-correction** 的加分测试。

---

## 1) 我已经完成了什么（总结版）

### 核心主线：Property Retrieval（属性检索）
已完成两个属性的 **baseline（不使用工具）** vs **tool-grounded（用 MP 工具取值后再让 LLM 复述）** 对比实验：

- **band_gap (eV)**：baseline + grounded
- **formation_energy_per_atom (eV/atom)**：baseline + grounded

### 补充测试（加分项）
- **Missing-field / Abstain 测试**：故意删除 record 中的 band_gap，看模型会不会“硬编”（hallucination）还是“拒答/unknown”
- **High-order retrieval (H1)**：小规模高阶组合/多步任务测试
- **Self-correction**：小规模自我纠错测试

### 主要现象（从已跑结果观察）
- baseline 下模型会出现 **“模式猜测/输出集中在少数几个常见数值”**（导致准确率很低）
- tool-grounded 后 **准确率显著提升**，并且与 tool 值几乎完全一致（mismatch≈0）

---

## 2) 文件夹结构说明

### 关键目录
- `logs/`  
  每次运行的 JSONL 日志（最重要，便于复现/统计）：
  - `*_baseline_band_gap.jsonl`
  - `*_baseline_fe.jsonl`
  - `*_grounded_band_gap.jsonl`
  - `*_grounded_fe.jsonl`

- `results/`  
  high-order / self-correction 的结果：
  - `*_high_order_H1.csv`
  - `*_high_order_self_correct.csv`

- `annotation/`  
  待人工标注的样本：
  - `20251224_103747_to_annotate.csv`

- `prompts/`
  - `templates.yaml`（自然语言 query 模版）

- `data/`
  - 预留给固定数据集/中间数据（部分数据也可能直接在根目录 CSV 中）

### Notebook / 脚本（入口文件）
- `day1_setup.ipynb`  
  MP 下载、筛选、采样、模板准备等
- `property_hallu_phase1_4.ipynb`  
  “Phase1–Phase4”风格的一体化流程（主 Notebook）
- `material_id.ipynb`  
  MPID 相关实验/辅助
- `make_queries.py`  
  生成 query 的辅助脚本

---

## 3) 关键数据与输出文件

### Ground Truth / 固定样本
- `gt_100.csv`：100 条样本的 GT
- `gt_200.csv`：200 条样本的 GT（主实验主要用这一份）
- `gt_200_full.csv`：更全字段的 GT（如果存在）
- `gt_fe.csv`：formation energy 对应 GT
- `ground_truth.csv`：整合型 GT（如果流程中使用）

### Baseline（不使用工具）结果
- `baseline_band_gap_200.csv`
- `baseline_band_gap_200_labeled.csv`
- `baseline_fe_200.csv`
- `baseline_fe_200_labeled_step3.csv`（包含更细的错误分类，比如 mode_guess）

小规模试跑：
- `baseline_band_gap_30.csv`
- `pilot_10_intrinsic_hallu.csv`
- `pilot_missing_band_gap_10.csv`
- `missing_band_gap_100.csv`

### Tool-grounded（工具取值后让 LLM 复述）
- `tool_grounded_band_gap_200.csv`
- formation energy 的 grounded 结果主要在 `logs/*grounded_fe*` 中，或由 notebook 导出对应 CSV

### Case studies（论文展示用例）
- `baseline_case_studies_20.csv`
- `fe_case_studies_10.csv`
- `case_studies_baseline_top10.md`

---

## 4) 实验设置（概念解释）

### Setting A：Baseline（no_tool）
直接问 LLM：
- 例：“What is the band gap (eV) of material mp-XXXX? Answer with a single number.”

评估方式：
- 从输出里解析第一个浮点数
- 与 MP Ground Truth 比对（设 tolerance，例如 0.05）

### Setting B：Tool-grounded
先用 MP API 拿到工具值，再让 LLM **只负责复述**：
- 例：“Given: band_gap = X. Output one sentence with the value.”

可选后验校验（post-check）：
- 如果 LLM 输出不等于 tool 值 → retry 或直接用 tool 值覆盖

---

## 5) 复现方式（最简）

### 环境要求
- Python 3.10+ 建议
- mp_api / pymatgen / pandas / numpy 等
- 环境变量中设置 MP key：
  - `MP_API_KEY` 或 `PMG_MAPI_KEY`

示例：
```bash
export MP_API_KEY="YOUR_KEY"