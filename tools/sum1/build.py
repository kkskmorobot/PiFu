from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(r"E:\App\DaiMa\JetBrains\Project\PiFuV2")
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    (16, 42, "BLG-MDA initial", "exploratory", "16_YOLO26m_Binary8Class_BinaryLesionGuidedAttention"),
    (18, 42, "Plain", "main", "18_YOLO26m_Binary8Class_Plain_DelayedUtilityEarlyStop"),
    (19, 42, "BLG-MDA delayed", "exploratory", "19_YOLO26m_Binary8Class_BLGAttention_DelayedUtilityEarlyStop"),
    (20, 42, "BLG-MDA adaptive", "exploratory", "20_YOLO26m_Binary8Class_AdaptiveResidualBLGAttention"),
    (21, 42, "ECA", "main", "21_YOLO26m_Binary8Class_P3P4_ECAOnlyAttention"),
    (22, 43, "Plain", "main", "22_YOLO26m_Binary8Class_Plain_Seed43Replication"),
    (23, 43, "ECA", "main", "23_YOLO26m_Binary8Class_P3P4_ECAOnlyAttention_Seed43Replication"),
    (24, 44, "Plain", "main", "24_YOLO26m_Binary8Class_Plain_Seed44Replication"),
    (25, 44, "ECA", "main", "25_YOLO26m_Binary8Class_P3P4_ECAOnlyAttention_Seed44Replication"),
    (26, 44, "ECA+Spatial7", "main", "26_YOLO26m_Binary8Class_P3P4_ECASpatialAttention_Seed44"),
    (27, 42, "ECA+Spatial7", "main", "27_YOLO26m_Binary8Class_P3P4_ECASpatialAttention_Seed42"),
    (28, 43, "ECA+Spatial7", "main", "28_YOLO26m_Binary8Class_P3P4_ECASpatialAttention_Seed43"),
    (29, 43, "Malignant group aux", "risk", "29_YOLO26m_Binary8Class_Plain_MalignantGroupAux_Seed43"),
    (30, 43, "MEL-NV focal", "risk", "30_YOLO26m_Binary8Class_Plain_MELNVFocalRisk_Seed43"),
    (31, 43, "MEL-NV aux head", "risk", "31_YOLO26m_Binary8Class_Plain_MELNVAuxHead_Seed43"),
    (32, 43, "Malignant aux head", "risk", "32_YOLO26m_Binary8Class_Plain_MalignantAuxHead_Seed43"),
    (33, 43, "CBAM", "attention_single", "33_YOLO26m_Binary8Class_P3P4_StandardCBAM_Seed43"),
]

METRIC_COLS = {
    "binary_iou": "metrics/binary_lesion_iou",
    "image_macro_f1": "metrics/image_macro_f1",
    "group_macro_f1": "metrics/group_macro_f1",
    "group_recall": "metrics/group_macro_recall",
    "mel_to_nv": "metrics/group_mel_to_nv_rate",
    "malignant_to_benign": "metrics/group_malignant_to_benign_rate",
    "nll": "metrics/group_nll",
    "brier": "metrics/group_brier",
    "ece": "metrics/group_ece",
    "joint": "metrics/joint_score",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_results(exp_name: str) -> Path:
    matches = list((PROJECT / "experiments" / exp_name).glob("results/*/results.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one results.csv for {exp_name}, found {len(matches)}")
    return matches[0]


rows: list[dict] = []
sources: list[dict] = []
for run, seed, family, role, exp_name in RUNS:
    path = locate_results(exp_name)
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    required = ["epoch", *METRIC_COLS.values()]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Run{run} missing columns: {missing}")
    numeric = frame[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError(f"Run{run} has non-finite required metrics")
    order = frame.assign(
        __joint=numeric[METRIC_COLS["joint"]],
        __f1=numeric[METRIC_COLS["group_macro_f1"]],
        __iou=numeric[METRIC_COLS["binary_iou"]],
        __epoch=numeric["epoch"],
    ).sort_values(["__joint", "__f1", "__iou", "__epoch"], ascending=[False, False, False, True])
    best = order.iloc[0]
    record = {
        "run": run,
        "seed": seed,
        "family": family,
        "role": role,
        "best_epoch": int(best["epoch"]),
        "epochs_recorded": int(len(frame)),
        "source": str(path),
    }
    for out_name, col in METRIC_COLS.items():
        record[out_name] = float(best[col])
    record["risk_feasible"] = bool(record["mel_to_nv"] <= 0.1077844311)
    rows.append(record)
    sources.append({"kind": "results_csv", "run": run, "path": str(path), "sha256": sha256(path), "rows": len(frame)})

runs_df = pd.DataFrame(rows).sort_values("run")
runs_df.to_csv(OUT / "runs.csv", index=False, float_format="%.8f")

main_df = runs_df[runs_df["role"] == "main"].copy()
families = ["Plain", "ECA", "ECA+Spatial7"]
metrics = list(METRIC_COLS)
family_rows: list[dict] = []
for family in families:
    block = main_df[main_df["family"] == family].sort_values("seed")
    if list(block["seed"]) != [42, 43, 44]:
        raise RuntimeError(f"Incomplete three-seed family: {family}")
    out = {"family": family, "n": 3}
    for metric in metrics:
        out[f"{metric}_mean"] = float(block[metric].mean())
        out[f"{metric}_sd"] = float(block[metric].std(ddof=1))
    out["feasible_count"] = int(block["risk_feasible"].sum())
    family_rows.append(out)
family_df = pd.DataFrame(family_rows)
family_df.to_csv(OUT / "families.csv", index=False, float_format="%.8f")

complexity_specs = [
    ("Plain", 18, PROJECT / "tools" / "run18" / "RUN18_PREFLIGHT_REPORT.json", 14507338),
    ("ECA", 21, PROJECT / "tools" / "run21" / "RUN21_PREFLIGHT_REPORT.json", 14507348),
    ("ECA+Spatial7", 26, PROJECT / "tools" / "run26" / "RUN26_PREFLIGHT_REPORT.json", 14507544),
    ("CBAM", 33, PROJECT / "tools" / "run33" / "RUN33_PREFLIGHT_REPORT.json", 14548494),
]
complexity_rows = []
plain_params = complexity_specs[0][3]
for family, source_run, report_path, expected_params in complexity_specs:
    report_text = report_path.read_text(encoding="utf-8")
    if f'"parameters": {expected_params}' not in report_text:
        raise RuntimeError(f"Parameter count mismatch in {report_path}")
    complexity_rows.append({
        "family": family,
        "source_run": source_run,
        "parameters": expected_params,
        "delta_vs_plain": expected_params - plain_params,
        "relative_increase_percent": 100.0 * (expected_params - plain_params) / plain_params,
        "gflops_note": "59.1 semantic backbone reported; custom head total GFLOPs not uniformly recorded",
    })
    sources.append({"kind": "preflight_complexity", "run": source_run, "path": str(report_path), "sha256": sha256(report_path), "rows": 1})
complexity_df = pd.DataFrame(complexity_rows)
complexity_df.to_csv(OUT / "complexity.csv", index=False, float_format="%.8f")

plain = main_df[main_df["family"] == "Plain"].set_index("seed")
paired_rows: list[dict] = []
for family in ["ECA", "ECA+Spatial7"]:
    candidate = main_df[main_df["family"] == family].set_index("seed")
    for seed in [42, 43, 44]:
        out = {"comparison": f"{family} - Plain", "seed": seed}
        for metric in metrics:
            out[metric] = float(candidate.loc[seed, metric] - plain.loc[seed, metric])
        paired_rows.append(out)
    mean_row = {"comparison": f"{family} - Plain", "seed": "mean"}
    for metric in metrics:
        mean_row[metric] = float(np.mean([r[metric] for r in paired_rows if r["comparison"] == mean_row["comparison"]]))
    paired_rows.append(mean_row)
paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(OUT / "paired.csv", index=False, float_format="%.8f")

val_result = PROJECT / "experiments" / "34_YOLO26m_Plain_FixedCostDecision_ThreeSeed_DevelopmentOnly" / "results" / "run34_plain_fixed_cost_decision_three_seed_val" / "RUN34_RESULT.json"
val = json.loads(val_result.read_text(encoding="utf-8"))
if val["status"] != "COMPLETE" or val["acceptance_gate"]["passed"]:
    raise RuntimeError("Val1 completion/acceptance state is unexpected")
if any(val[k] for k in ("training_performed", "test_metrics_used", "external_metrics_used", "pad_ufes_20_used")):
    raise RuntimeError("Val1 boundary flag violation")
sources.append({"kind": "val1_result", "run": "Val1", "path": str(val_result), "sha256": sha256(val_result), "rows": len(val["per_seed"])})

val_rows: list[dict] = []
all_group_frames: list[pd.DataFrame] = []
for item in val["per_seed"]:
    row = {"seed": item["seed"], "source_run": item["run"], "binary_iou": item["binary_iou"]}
    for prefix in ("argmax", "fixed_cost"):
        group = item[prefix]["group"]
        row.update({
            f"{prefix}_group_macro_f1": group["macro_f1"],
            f"{prefix}_group_recall": group["macro_recall"],
            f"{prefix}_mel_to_nv": group["mel_to_nv_rate"],
            f"{prefix}_malignant_to_benign": group["malignant_to_benign_rate"],
            f"{prefix}_joint": item[prefix]["joint"],
        })
    row.update({f"delta_{k}": v for k, v in item["delta_fixed_cost_minus_argmax"].items()})
    val_rows.append(row)
    group_csv = val_result.parent / f"seed{item['seed']}_run{item['run']}" / "group_predictions.csv"
    g = pd.read_csv(group_csv)
    if len(g) != 587:
        raise RuntimeError(f"Unexpected Val1 group count: {group_csv}")
    all_group_frames.append(g.assign(seed=item["seed"]))
    sources.append({"kind": "val1_group_predictions", "run": item["run"], "path": str(group_csv), "sha256": sha256(group_csv), "rows": len(g)})

val_df = pd.DataFrame(val_rows).sort_values("seed")
val_df.to_csv(OUT / "val1.csv", index=False, float_format="%.8f")
groups = pd.concat(all_group_frames, ignore_index=True)
classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc"]
for col in [f"score_{c}" for c in classes]:
    groups[col] = pd.to_numeric(groups[col], errors="raise")

# Attention comparison figure.
utility_metrics = ["binary_iou", "image_macro_f1", "group_macro_f1", "group_recall", "joint"]
risk_metrics = ["mel_to_nv", "malignant_to_benign"]
colors = ["#4C78A8", "#F58518", "#54A24B"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
x = np.arange(len(utility_metrics))
width = 0.24
for i, family in enumerate(families):
    rec = family_df[family_df["family"] == family].iloc[0]
    axes[0].bar(x + (i - 1) * width, [rec[f"{m}_mean"] for m in utility_metrics], width,
                yerr=[rec[f"{m}_sd"] for m in utility_metrics], capsize=3, label=family, color=colors[i])
axes[0].set_xticks(x, ["IoU", "Image F1", "Group F1", "Group recall", "Joint"], rotation=20, ha="right")
axes[0].set_ylim(0.45, 0.82)
axes[0].set_ylabel("Development metric")
axes[0].set_title("Three-seed utility")
axes[0].grid(axis="y", alpha=0.2)
axes[0].legend(frameon=False)
x2 = np.arange(len(risk_metrics))
for i, family in enumerate(families):
    rec = family_df[family_df["family"] == family].iloc[0]
    axes[1].bar(x2 + (i - 1) * width, [rec[f"{m}_mean"] for m in risk_metrics], width,
                yerr=[rec[f"{m}_sd"] for m in risk_metrics], capsize=3, label=family, color=colors[i])
axes[1].axhline(0.1077844311, color="#B22222", linestyle="--", linewidth=1.3, label="Feasibility limit")
axes[1].set_xticks(x2, ["MEL to NV", "Malignant to benign"])
axes[1].set_ylim(0, 0.55)
axes[1].set_ylabel("Group error rate")
axes[1].set_title("Three-seed risk")
axes[1].grid(axis="y", alpha=0.2)
axes[1].legend(frameon=False)
fig.savefig(OUT / "attention.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Aggregate normalized group confusion matrix for the locked argmax rule.
cm = pd.crosstab(groups["target"].astype(int), groups["argmax_id"].astype(int)).reindex(index=range(8), columns=range(8), fill_value=0).to_numpy()
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.3, 6.2), constrained_layout=True)
im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
for i in range(8):
    for j in range(8):
        val_cell = cm_norm[i, j]
        ax.text(j, i, f"{val_cell:.2f}", ha="center", va="center", fontsize=8,
                color="white" if val_cell > 0.55 else "#222222")
ax.set_xticks(range(8), classes, rotation=35, ha="right")
ax.set_yticks(range(8), classes)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Locked argmax group confusion (three seeds)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")
fig.savefig(OUT / "confusion.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Calibration plot from frozen group probabilities.
score_cols = [f"score_{c}" for c in classes]
fig, ax = plt.subplots(figsize=(6.5, 5.2), constrained_layout=True)
bins = np.linspace(0, 1, 11)
calibration_rows = []
for seed, color in zip([42, 43, 44], colors):
    block = groups[groups["seed"] == seed]
    scores = block[score_cols].to_numpy(dtype=float)
    confidence = scores.max(axis=1)
    correct = (block["argmax_id"].to_numpy(dtype=int) == block["target"].to_numpy(dtype=int)).astype(float)
    ids = np.clip(np.digitize(confidence, bins, right=True) - 1, 0, 9)
    xs, ys = [], []
    for b in range(10):
        mask = ids == b
        if mask.any():
            xs.append(float(confidence[mask].mean()))
            ys.append(float(correct[mask].mean()))
            calibration_rows.append({"seed": seed, "bin": b + 1, "count": int(mask.sum()), "mean_confidence": xs[-1], "accuracy": ys[-1]})
    ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=f"seed{seed}")
ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.2, label="Perfect calibration")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean confidence")
ax.set_ylabel("Observed accuracy")
ax.set_title("Group calibration on development data")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
fig.savefig(OUT / "calibration.png", dpi=220, bbox_inches="tight")
plt.close(fig)
pd.DataFrame(calibration_rows).to_csv(OUT / "calibration.csv", index=False, float_format="%.8f")

# Val1 trade-off figure. Negative means fixed-cost is lower than argmax.
delta_names = ["group_macro_f1", "group_recall", "mel_to_nv", "malignant_to_benign", "joint"]
mean_delta = val["acceptance_gate"]["mean_delta"]
fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
vals = [mean_delta[m] for m in delta_names]
bar_colors = ["#D9534F", "#D9534F", "#2E8B57", "#2E8B57", "#D9534F"]
bars = ax.bar(np.arange(len(vals)), vals, color=bar_colors)
ax.axhline(0, color="#333333", linewidth=1)
ax.set_xticks(np.arange(len(vals)), ["Group F1", "Group recall", "MEL to NV", "Malignant to benign", "Joint"], rotation=20, ha="right")
ax.set_ylabel("Fixed cost minus argmax")
ax.set_title("Val1 mean decision-rule trade-off")
ax.grid(axis="y", alpha=0.2)
for bar, value in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008 if value < 0 else value - 0.008,
            f"{value:.3f}", ha="center", va="bottom" if value < 0 else "top", fontsize=9,
            color="white", fontweight="bold")
fig.savefig(OUT / "val1.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Segmentation example montage from the locked Run22 representative.
run22_dir = locate_results("22_YOLO26m_Binary8Class_Plain_Seed43Replication").parent
panels = [
    (run22_dir / "val_batch0_labels.jpg", "Ground truth batch 1"),
    (run22_dir / "val_batch0_pred.jpg", "Prediction batch 1"),
    (run22_dir / "val_batch1_labels.jpg", "Ground truth batch 2"),
    (run22_dir / "val_batch1_pred.jpg", "Prediction batch 2"),
]
loaded = [(Image.open(p).convert("RGB"), title) for p, title in panels]
cell_w = 720
cell_h = int(loaded[0][0].height * cell_w / loaded[0][0].width)
title_h = 34
canvas = Image.new("RGB", (cell_w * 2, (cell_h + title_h) * 2), "white")
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default()
for idx, (img, title) in enumerate(loaded):
    img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
    x0 = (idx % 2) * cell_w
    y0 = (idx // 2) * (cell_h + title_h)
    canvas.paste(img, (x0, y0 + title_h))
    draw.text((x0 + 8, y0 + 9), title, fill="#222222", font=font)
canvas.save(OUT / "segmentation.png", quality=94)

summary = {
    "status": "COMPLETE",
    "selection_basis": "maximum development joint score with the recorded tie-break order",
    "runs": rows,
    "family_summary": family_rows,
    "paired_deltas": paired_rows,
    "complexity": complexity_rows,
    "val1": val,
    "sources": sources,
    "boundaries": {"test_used": False, "external_used": False, "pad_ufes_20_used": False},
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")

plain_stats = family_df.set_index("family").loc["Plain"]
eca_stats = family_df.set_index("family").loc["ECA"]
spatial_stats = family_df.set_index("family").loc["ECA+Spatial7"]
md = f"""# YOLO26m development 汇总

`SUM1_STATUS=COMPLETE`
`TEST_METRICS_USED=FALSE`
`EXTERNAL_METRICS_USED=FALSE`
`PAD_UFES_20_USED=FALSE`

## 主结论

三 seed 的统一 best-Joint 比较支持 Plain YOLO26m P3/P4 partial-label 多任务架构作为最终研究候选。Plain 的平均 Joint 为 `{plain_stats['joint_mean']:.5f}`，ECA 为 `{eca_stats['joint_mean']:.5f}`，ECA+Spatial7 为 `{spatial_stats['joint_mean']:.5f}`。三组正式 feasible checkpoint 数均为 0。

ECA 与 ECA+Spatial7 没有形成稳定的总体效用增益。标准 CBAM、BLG-MDA 演进以及 Run29–Run32 风险训练分支均保留为探索性或失败消融证据，不进入最终架构。

Val1 的固定代价规则降低了两项恶性误判，但平均 Group Macro-F1 与 Joint 的损害超过预注册保护线，因此最终决策规则保持八类 argmax。

## 三 seed 主实验均值 ± 样本标准差

| 架构 | Binary IoU | Image Macro-F1 | Group Macro-F1 | Group Recall | MEL→NV | malignant→benign | Joint | feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
for _, r in family_df.iterrows():
    md += f"| {r['family']} | {r['binary_iou_mean']:.5f} ± {r['binary_iou_sd']:.5f} | {r['image_macro_f1_mean']:.5f} ± {r['image_macro_f1_sd']:.5f} | {r['group_macro_f1_mean']:.5f} ± {r['group_macro_f1_sd']:.5f} | {r['group_recall_mean']:.5f} ± {r['group_recall_sd']:.5f} | {r['mel_to_nv_mean']:.5f} ± {r['mel_to_nv_sd']:.5f} | {r['malignant_to_benign_mean']:.5f} ± {r['malignant_to_benign_sd']:.5f} | {r['joint_mean']:.5f} ± {r['joint_sd']:.5f} | {int(r['feasible_count'])}/3 |\n"
md += """

## 参数量

| 架构 | 参数量 | 相对 Plain 增量 | 相对增幅 |
|---|---:|---:|---:|
"""
for _, r in complexity_df.iterrows():
    md += f"| {r['family']} | {int(r['parameters']):,} | {int(r['delta_vs_plain']):+,} | {r['relative_increase_percent']:.4f}% |\n"
md += """

语义主干在现有验证输出中报告为约 `59.1 GFLOPs`；自定义分类/注意力头的总 GFLOPs 没有在所有 run 中以同一统计口径保存，因此不据此编造跨结构 GFLOPs 差值。

## 论文解释边界

- YOLO26s 只承担早期低成本筛选；YOLO26m 是正式主模型。
- Run18/22/24、Run21/23/25、Run27/28/26 构成三 seed 注意力主消融。
- Run33 是标准 CBAM 单 seed 对照；Run16/19/20 是 BLG-MDA 演进证据；Run29–Run32 是未进入复现阶段的风险分支。
- Val1 是冻结模型的 development-only 决策实验，历史预注册名为 Run34。
- 最终研究候选不等于临床风险合格模型；正式 feasible checkpoint 为 0。
- 本汇总未读取 IMA++ test、ISIC2018 external 或 PAD-UFES-20。
"""
(OUT / "summary.md").write_text(md, encoding="utf-8")

print("SUM1_DATA_AND_FIGURES=PASS")
print(f"RUN_ROWS={len(runs_df)}")
print(f"SOURCE_FILES={len(sources)}")
