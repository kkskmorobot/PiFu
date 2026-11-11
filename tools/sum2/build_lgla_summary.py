#!/usr/bin/env python3
"""Build the frozen LGLA-v1 development/test tables and thesis figures."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(r"E:\App\DaiMa\JetBrains\Project\PiFuV2")
OUT = Path(r"C:\Users\ROBOT\Documents\皮肤识别\lgla_finalize\generated")
OUT.mkdir(parents=True, exist_ok=True)

METRICS = {
    "binary_iou": "metrics/binary_lesion_iou",
    "binary_dice": "metrics/binary_lesion_dice",
    "pixel_accuracy": "metrics/pixel_acc",
    "image_macro_f1": "metrics/image_macro_f1",
    "image_macro_recall": "metrics/image_macro_recall",
    "group_macro_f1": "metrics/group_macro_f1",
    "group_macro_recall": "metrics/group_macro_recall",
    "mel_to_nv_rate": "metrics/group_mel_to_nv_rate",
    "malignant_to_benign_rate": "metrics/group_malignant_to_benign_rate",
    "nll": "metrics/group_nll",
    "brier": "metrics/group_brier",
    "ece": "metrics/group_ece",
    "joint_score": "metrics/joint_score",
}

RUNS = [
    ("Plain", 42, "Run18", 50, PROJECT / "experiments/18_YOLO26m_Binary8Class_Plain_DelayedUtilityEarlyStop/results/run18_yolo26m_plain_delayed_utility_early_stop_seed42/results.csv"),
    ("Plain", 43, "Run22", 60, PROJECT / "experiments/22_YOLO26m_Binary8Class_Plain_Seed43Replication/results/run22_yolo26m_plain_seed43_replication/results.csv"),
    ("Plain", 44, "Run24", 78, PROJECT / "experiments/24_YOLO26m_Binary8Class_Plain_Seed44Replication/results/run24_yolo26m_plain_seed44_replication/results.csv"),
    ("LGLA v1", 42, "LG2", 74, PROJECT / "experiments/lg2_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed42/results/lg2_yolo26m_lgla_v1_gate_balance_seed42/results.csv"),
    ("LGLA v1", 43, "LGX", 99, PROJECT / "experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43/results/lgx_yolo26m_lgla_v1_gate_balance_seed43/results.csv"),
    ("LGLA v1", 44, "LG3", 97, PROJECT / "experiments/lg3_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed44/results/lg3_yolo26m_lgla_v1_gate_balance_seed44/results.csv"),
    ("AB1 / unguided run", 43, "AB1", 68, PROJECT / "experiments/ab1_YOLO26m_Binary8Class_LGLAv1_Unguided_GateBalance_Seed43/results/ab1_yolo26m_lgla_v1_unguided_gate_balance_seed43/results.csv"),
    ("AB2 / fixed fusion", 43, "AB2", 49, PROJECT / "experiments/ab2_YOLO26m_Binary8Class_LGLAv1_FixedEqualFusion_Seed43/results/ab2_yolo26m_lgla_v1_fixed_equal_fusion_seed43/results.csv"),
]


def read_epoch(path: Path, visible_epoch: int) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(float(row["epoch"])) == visible_epoch]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one visible epoch {visible_epoch} in {path}, found {len(matches)}")
    return {name: float(matches[0][column]) for name, column in METRICS.items()}


records = []
for architecture, seed, run, epoch, source in RUNS:
    records.append({
        "architecture": architecture,
        "seed": seed,
        "run": run,
        "visible_epoch": epoch,
        "source": str(source),
        **read_epoch(source, epoch),
    })


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


plain = {row["seed"]: row for row in records if row["architecture"] == "Plain"}
lgla = {row["seed"]: row for row in records if row["architecture"] == "LGLA v1"}
paired_metrics = list(METRICS)
paired_rows = []
for seed in (42, 43, 44):
    row = {
        "seed": seed,
        "plain_run": plain[seed]["run"],
        "plain_visible_epoch": plain[seed]["visible_epoch"],
        "lgla_run": lgla[seed]["run"],
        "lgla_visible_epoch": lgla[seed]["visible_epoch"],
    }
    for metric in paired_metrics:
        row[f"plain_{metric}"] = plain[seed][metric]
        row[f"lgla_{metric}"] = lgla[seed][metric]
        row[f"delta_{metric}"] = lgla[seed][metric] - plain[seed][metric]
    paired_rows.append(row)

with (OUT / "paired_development.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
    writer.writeheader()
    writer.writerows(paired_rows)

summary_rows = []
for metric in paired_metrics:
    p_mean, p_sd = mean_sd([plain[s][metric] for s in (42, 43, 44)])
    l_mean, l_sd = mean_sd([lgla[s][metric] for s in (42, 43, 44)])
    deltas = [lgla[s][metric] - plain[s][metric] for s in (42, 43, 44)]
    d_mean, d_sd = mean_sd(deltas)
    higher_better = metric not in {"mel_to_nv_rate", "malignant_to_benign_rate", "nll", "brier", "ece"}
    improved_seeds = sum(delta > 0 if higher_better else delta < 0 for delta in deltas)
    summary_rows.append({
        "metric": metric,
        "direction": "higher_better" if higher_better else "lower_better",
        "plain_mean": p_mean,
        "plain_sample_sd": p_sd,
        "lgla_mean": l_mean,
        "lgla_sample_sd": l_sd,
        "paired_delta_mean_lgla_minus_plain": d_mean,
        "paired_delta_sample_sd": d_sd,
        "improved_seed_count_out_of_3": improved_seeds,
    })

with (OUT / "three_seed_development_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)

ablation_order = ["Plain", "AB1 / unguided run", "AB2 / fixed fusion", "LGLA v1"]
ablation_rows = []
for name in ablation_order:
    row = next(item for item in records if item["architecture"] == name and item["seed"] == 43)
    ablation_rows.append({key: row[key] for key in ("architecture", "run", "seed", "visible_epoch", *paired_metrics)})
with (OUT / "seed43_ablation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0]))
    writer.writeheader()
    writer.writerows(ablation_rows)

test1 = json.loads((PROJECT / "experiments/test1/results/result.json").read_text(encoding="utf-8"))
test2 = json.loads((PROJECT / "experiments/test2/results/result.json").read_text(encoding="utf-8"))
test_rows = []
sections = [
    ("internal_segmentation", ("binary_iou", "binary_dice", "pixel_accuracy")),
    ("restricted_classification", ("accuracy", "mel_recall", "nv_recall", "balanced_recall", "mel_to_nv_rate", "nv_to_mel_rate", "nll", "brier", "ece")),
    ("external_segmentation", ("binary_iou", "binary_dice", "pixel_accuracy")),
]
for section, metrics in sections:
    for metric in metrics:
        p = test1["aggregate"][section][metric]
        l = test2["aggregate"][section][metric]
        test_rows.append({
            "section": section,
            "metric": metric,
            "plain_mean": p["mean"],
            "plain_sample_sd": p["sample_sd"],
            "lgla_mean": l["mean"],
            "lgla_sample_sd": l["sample_sd"],
            "delta_lgla_minus_plain": l["mean"] - p["mean"],
        })
with (OUT / "test2_vs_test1.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
    writer.writeheader()
    writer.writerows(test_rows)

complexity_rows = [
    {"architecture": "Plain", "parameters": 14507338, "delta_parameters": 0, "relative_delta_percent": 0.0, "source": "tools/val1/RUN34_PREFLIGHT_REPORT.json"},
    {"architecture": "LGLA v1", "parameters": 14540748, "delta_parameters": 33410, "relative_delta_percent": (14540748 / 14507338 - 1) * 100, "source": "tools/lgx/LGX_PREFLIGHT_REPORT.json"},
    {"architecture": "AB1 / unguided run", "parameters": 14540748, "delta_parameters": 33410, "relative_delta_percent": (14540748 / 14507338 - 1) * 100, "source": "tools/ab1/AB1_PREFLIGHT_REPORT.json"},
    {"architecture": "AB2 / fixed fusion", "parameters": 14540748, "delta_parameters": 33410, "relative_delta_percent": (14540748 / 14507338 - 1) * 100, "source": "tools/ab2/AB2_PREFLIGHT_REPORT.json"},
]
with (OUT / "complexity.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(complexity_rows[0]))
    writer.writeheader()
    writer.writerows(complexity_rows)

payload = {
    "status": "COMPLETE_FROZEN_BEFORE_THESIS_WRITING",
    "development": {"paired_rows": paired_rows, "summary": summary_rows, "ablation_seed43": ablation_rows},
    "test": {"comparison": test_rows, "test2_fingerprint": test2["fingerprint"]},
    "complexity": complexity_rows,
    "conclusion": {
        "development": "LGLA v1 improves the paired three-seed means of Binary IoU, Image Macro-F1, Group Macro-F1, and Joint; Image/Group/Joint improve in all three seeds.",
        "ablation": "At seed43, full LGLA v1 exceeds both the unguided run and fixed-equal-fusion ablation on Group Macro-F1 and Joint, supporting both lesion guidance and learned fusion.",
        "test": "Internal Test2 segmentation is essentially tied with Plain Test1 and restricted accuracy is slightly higher, while MEL recall/calibration and incremental external segmentation are worse.",
        "scope": "The evidence supports a development-set utility improvement, not a universal clinical/generalization improvement.",
    },
    "hashes": {
        "test1_result_sha256": hashlib.sha256((PROJECT / "experiments/test1/results/result.json").read_bytes()).hexdigest(),
        "test2_result_sha256": hashlib.sha256((PROJECT / "experiments/test2/results/result.json").read_bytes()).hexdigest(),
    },
}
(OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
colors = {"Plain": "#64748b", "LGLA v1": "#2563eb", "AB1 / unguided run": "#f59e0b", "AB2 / fixed fusion": "#8b5cf6"}

core = ["binary_iou", "image_macro_f1", "group_macro_f1", "joint_score"]
plain_means = [next(r["plain_mean"] for r in summary_rows if r["metric"] == m) for m in core]
lgla_means = [next(r["lgla_mean"] for r in summary_rows if r["metric"] == m) for m in core]
x = np.arange(len(core)); width = 0.36
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.bar(x - width / 2, plain_means, width, label="Plain", color=colors["Plain"])
ax.bar(x + width / 2, lgla_means, width, label="LGLA v1", color=colors["LGLA v1"])
ax.set_xticks(x, ["Binary IoU", "Image Macro-F1", "Group Macro-F1", "Joint"])
ax.set_ylim(0.5, 0.8); ax.set_ylabel("Three-seed development mean"); ax.legend(frameon=False)
ax.set_title("Development core metrics: Plain vs LGLA v1")
fig.tight_layout(); fig.savefig(OUT / "figure_dev_core_means.png", bbox_inches="tight"); plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(11, 3.5), sharex=True)
for ax, metric, title in zip(axes, core, ["Binary IoU", "Image F1", "Group F1", "Joint"]):
    vals = [lgla[s][metric] - plain[s][metric] for s in (42, 43, 44)]
    ax.axhline(0, color="#334155", lw=0.8); ax.bar(["42", "43", "44"], vals, color=[colors["LGLA v1"] if v >= 0 else "#dc2626" for v in vals])
    ax.set_title(title); ax.set_xlabel("seed")
axes[0].set_ylabel("LGLA - Plain")
fig.suptitle("Paired development deltas by seed", y=1.02); fig.tight_layout(); fig.savefig(OUT / "figure_paired_seed_deltas.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 4.6))
names = [r["architecture"] for r in ablation_rows]
group = [r["group_macro_f1"] for r in ablation_rows]
joint = [r["joint_score"] for r in ablation_rows]
x = np.arange(len(names)); width = 0.36
ax.bar(x - width / 2, group, width, label="Group Macro-F1", color="#0f766e")
ax.bar(x + width / 2, joint, width, label="Joint", color="#2563eb")
ax.set_xticks(x, ["Plain", "AB1 unguided", "AB2 fixed", "Full LGLA"]); ax.set_ylim(0.55, 0.74)
ax.set_title("Seed43 ablation comparison"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "figure_seed43_ablation.png", bbox_inches="tight"); plt.close(fig)

test_metrics = [("internal_segmentation", "binary_iou", "Internal IoU"), ("restricted_classification", "accuracy", "Accuracy"), ("restricted_classification", "balanced_recall", "Balanced recall"), ("external_segmentation", "binary_iou", "External IoU")]
pvals = [next(r["plain_mean"] for r in test_rows if r["section"] == s and r["metric"] == m) for s, m, _ in test_metrics]
lvals = [next(r["lgla_mean"] for r in test_rows if r["section"] == s and r["metric"] == m) for s, m, _ in test_metrics]
fig, ax = plt.subplots(figsize=(8.2, 4.6)); x = np.arange(len(test_metrics))
ax.bar(x - width / 2, pvals, width, label="Plain Test1", color=colors["Plain"])
ax.bar(x + width / 2, lvals, width, label="LGLA Test2", color=colors["LGLA v1"])
ax.set_xticks(x, [t for _, _, t in test_metrics]); ax.set_ylim(0.4, 0.85); ax.set_title("Frozen test comparison")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "figure_test_comparison.png", bbox_inches="tight"); plt.close(fig)

diag_paths = [
    PROJECT / "experiments/lg2_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed42/results/lg2_yolo26m_lgla_v1_gate_balance_seed42/analysis/val_attention_diagnostics_v1/LG2_VAL_ATTENTION_DIAGNOSTICS.json",
    PROJECT / "experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43/results/lgx_yolo26m_lgla_v1_gate_balance_seed43/analysis/val_attention_diagnostics_v1/LGX_VAL_ATTENTION_DIAGNOSTICS.json",
    PROJECT / "experiments/lg3_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed44/results/lg3_yolo26m_lgla_v1_gate_balance_seed44/analysis/val_attention_diagnostics_v1/LG3_VAL_ATTENTION_DIAGNOSTICS.json",
]
diags = [json.loads(p.read_text(encoding="utf-8")) for p in diag_paths]
box_stats = []
for seed, diag in zip((42, 43, 44), diags):
    d = diag["diagnostics"]["local_gate"]
    box_stats.append({"label": f"seed{seed}", "whislo": d["p05"], "q1": d["p25"], "med": d["median"], "q3": d["p75"], "whishi": d["p95"], "mean": d["mean"], "fliers": []})
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.bxp(box_stats, showmeans=True, meanline=True, patch_artist=True, boxprops={"facecolor": "#DBEAFE", "edgecolor": "#2563EB"}, medianprops={"color": "#1E3A8A"})
ax.axhline(0.5, color="#64748B", ls="--", lw=1); ax.set_ylim(0, 1); ax.set_ylabel("Local branch gate weight")
ax.set_title("Frozen validation gate distribution (5th–95th percentiles)")
fig.tight_layout(); fig.savefig(OUT / "figure_gate_weight_distribution.png", bbox_inches="tight"); plt.close(fig)

print(json.dumps({"status": "PASS", "output": str(OUT), "files": sorted(p.name for p in OUT.iterdir())}, ensure_ascii=False, indent=2))
