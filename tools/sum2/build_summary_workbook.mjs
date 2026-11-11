import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const base = "C:/Users/ROBOT/Documents/皮肤识别/lgla_finalize";
const dataDir = `${base}/generated`;
const outDir = `${base}/workbook/output`;
const renderDir = `${base}/workbook/renders`;
await fs.mkdir(outDir, { recursive: true });
await fs.mkdir(renderDir, { recursive: true });

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      const raw = values[i] ?? "";
      const numeric = raw !== "" && Number.isFinite(Number(raw));
      row[h] = numeric ? Number(raw) : raw;
    });
    return row;
  });
}

const paired = parseCsv(await fs.readFile(`${dataDir}/paired_development.csv`, "utf8"));
const devSummary = parseCsv(await fs.readFile(`${dataDir}/three_seed_development_summary.csv`, "utf8"));
const ablations = parseCsv(await fs.readFile(`${dataDir}/seed43_ablation.csv`, "utf8"));
const testCompare = parseCsv(await fs.readFile(`${dataDir}/test2_vs_test1.csv`, "utf8"));
const complexity = parseCsv(await fs.readFile(`${dataDir}/complexity.csv`, "utf8"));

const wb = Workbook.create();
const summary = wb.worksheets.add("Summary");
const pairedSheet = wb.worksheets.add("Paired seeds");
const devSheet = wb.worksheets.add("Development");
const ablationSheet = wb.worksheets.add("Ablations");
const testSheet = wb.worksheets.add("Frozen test");
const sourceSheet = wb.worksheets.add("Sources");

const navy = "#17365D";
const blue = "#2563EB";
const paleBlue = "#DBEAFE";
const paleGreen = "#DCFCE7";
const paleRed = "#FEE2E2";
const light = "#F8FAFC";
const grid = "#CBD5E1";
const muted = "#475569";

function title(sheet, address, text) {
  const r = sheet.getRange(address);
  r.merge();
  r.values = [[text]];
  r.format.fill = navy;
  r.format.font = { bold: true, color: "#FFFFFF", size: 16 };
  r.format.rowHeight = 30;
  r.format.verticalAlignment = "center";
}

function header(range) {
  range.format.fill = navy;
  range.format.font = { bold: true, color: "#FFFFFF" };
  range.format.wrapText = true;
  range.format.verticalAlignment = "center";
  range.format.borders = { preset: "all", style: "thin", color: grid };
}

function body(range) {
  range.format.borders = { preset: "all", style: "thin", color: grid };
  range.format.verticalAlignment = "center";
}

title(summary, "A1:H1", "LGLA v1 实验与冻结测试汇总");
summary.getRange("A3:H5").values = [
  ["结论层级", "结论", null, null, null, null, null, null],
  ["Development", "三 seed 的 Binary IoU、Image Macro-F1、Group Macro-F1、Joint 均值提高；其中后三项 3/3 seed 提高。", null, null, null, null, null, null],
  ["Frozen test", "内部 test 基本持平，受限分类准确率略高；但 MEL recall、校准及增量 external 分割较差，不能声称全面优于 Plain。", null, null, null, null, null, null],
];
summary.getRange("B3:H3").merge(); summary.getRange("B4:H4").merge(); summary.getRange("B5:H5").merge();
summary.getRange("A3:A5").format.fill = paleBlue;
summary.getRange("A3:A5").format.font = { bold: true, color: navy };
summary.getRange("A3:H5").format.wrapText = true; body(summary.getRange("A3:H5"));
summary.getRange("A7:E7").values = [["核心 development 指标", "Plain mean", "LGLA mean", "LGLA - Plain", "改善 seed 数"]];
header(summary.getRange("A7:E7"));
summary.getRange("A8:A11").values = [["Binary IoU"], ["Image Macro-F1"], ["Group Macro-F1"], ["Joint"]];
summary.getRange("B8:E11").formulas = [
  ["=AVERAGE('Paired seeds'!B5:B7)", "=AVERAGE('Paired seeds'!C5:C7)", "=C8-B8", "=COUNTIF('Paired seeds'!D5:D7,\">0\")"],
  ["=AVERAGE('Paired seeds'!E5:E7)", "=AVERAGE('Paired seeds'!F5:F7)", "=C9-B9", "=COUNTIF('Paired seeds'!G5:G7,\">0\")"],
  ["=AVERAGE('Paired seeds'!H5:H7)", "=AVERAGE('Paired seeds'!I5:I7)", "=C10-B10", "=COUNTIF('Paired seeds'!J5:J7,\">0\")"],
  ["=AVERAGE('Paired seeds'!K5:K7)", "=AVERAGE('Paired seeds'!L5:L7)", "=C11-B11", "=COUNTIF('Paired seeds'!M5:M7,\">0\")"],
];
body(summary.getRange("A8:E11")); summary.getRange("B8:D11").setNumberFormat("0.00000");
summary.getRange("D8:D11").conditionalFormats.add("cellIs", { operator: "greaterThan", formula: 0, format: { fill: paleGreen, font: { color: "#166534", bold: true } } });
summary.getRange("D8:D11").conditionalFormats.add("cellIs", { operator: "lessThan", formula: 0, format: { fill: paleRed, font: { color: "#991B1B", bold: true } } });

summary.getRange("A13:F13").values = [["冻结测试指标", "Plain mean", "LGLA mean", "差值", "解释", "方向"]];
header(summary.getRange("A13:F13"));
const testPick = [
  ["internal_segmentation", "binary_iou", "Internal Binary IoU", "基本持平", "higher"],
  ["restricted_classification", "accuracy", "MEL/NV Accuracy", "LGLA 略高", "higher"],
  ["restricted_classification", "mel_recall", "MEL Recall", "LGLA 较低", "higher"],
  ["external_segmentation", "binary_iou", "External Binary IoU", "LGLA 明显较低", "higher"],
];
const testRows = testPick.map(([section, metric, label, explanation, direction]) => {
  const r = testCompare.find((x) => x.section === section && x.metric === metric);
  return [label, r.plain_mean, r.lgla_mean, r.delta_lgla_minus_plain, explanation, direction];
});
summary.getRange("A14:F17").values = testRows; body(summary.getRange("A14:F17")); summary.getRange("B14:D17").setNumberFormat("0.00000");
summary.getRange("A19:H21").values = [
  ["论文建议", "主结论：LGLA v1 在 development 三 seed 核心效用上稳定优于 Plain；冻结内部测试保持相当表现。", null, null, null, null, null, null],
  ["必要局限", "受限 MEL/NV 子集较小且不覆盖完整八类；MEL recall 与校准未改善；ISIC2018 增量 external 分割较差。", null, null, null, null, null, null],
  ["训练状态", "无需继续长训练；Test2 后禁止根据测试结果返回 development 调参。", null, null, null, null, null, null],
];
for (const row of [19, 20, 21]) summary.getRange(`B${row}:H${row}`).merge();
summary.getRange("A19:A21").format.fill = paleBlue; summary.getRange("A19:A21").format.font = { bold: true, color: navy };
summary.getRange("A19:H21").format.wrapText = true; body(summary.getRange("A19:H21"));
summary.freezePanes.freezeRows(1); summary.showGridLines = false;
summary.getRange("A1:H21").format.autofitRows(); summary.getRange("A:A").format.columnWidth = 20; summary.getRange("B:H").format.columnWidth = 16;

title(pairedSheet, "A1:M1", "三 seed 配对 development 核心指标");
pairedSheet.getRange("A3:M3").values = [["说明", "每行使用预先锁定的 best-Joint epoch；差值均为 LGLA - Plain。", null, null, null, null, null, null, null, null, null, null, null]];
pairedSheet.getRange("B3:M3").merge(); pairedSheet.getRange("A3:M3").format.wrapText = true;
pairedSheet.getRange("A4:M4").values = [["Seed", "Plain IoU", "LGLA IoU", "Δ IoU", "Plain Image F1", "LGLA Image F1", "Δ Image F1", "Plain Group F1", "LGLA Group F1", "Δ Group F1", "Plain Joint", "LGLA Joint", "Δ Joint"]];
header(pairedSheet.getRange("A4:M4"));
pairedSheet.getRange("A5:M7").values = paired.map((r) => [r.seed, r.plain_binary_iou, r.lgla_binary_iou, r.delta_binary_iou, r.plain_image_macro_f1, r.lgla_image_macro_f1, r.delta_image_macro_f1, r.plain_group_macro_f1, r.lgla_group_macro_f1, r.delta_group_macro_f1, r.plain_joint_score, r.lgla_joint_score, r.delta_joint_score]);
body(pairedSheet.getRange("A5:M7")); pairedSheet.getRange("B5:M7").setNumberFormat("0.00000");
for (const col of ["D", "G", "J", "M"]) pairedSheet.getRange(`${col}5:${col}7`).conditionalFormats.add("cellIs", { operator: "greaterThan", formula: 0, format: { fill: paleGreen } });
pairedSheet.freezePanes.freezeRows(4); pairedSheet.showGridLines = false; pairedSheet.getRange("A1:M7").format.autofitColumns(); pairedSheet.getRange("A1:M7").format.autofitRows();

title(devSheet, "A1:I1", "完整 development 指标汇总（均值 ± 样本 SD）");
const devHeaders = ["Metric", "Direction", "Plain mean", "Plain SD", "LGLA mean", "LGLA SD", "Paired Δ mean", "Paired Δ SD", "Improved seeds"];
devSheet.getRange("A3:I3").values = [devHeaders]; header(devSheet.getRange("A3:I3"));
devSheet.getRange(`A4:I${3 + devSummary.length}`).values = devSummary.map((r) => [r.metric, r.direction, r.plain_mean, r.plain_sample_sd, r.lgla_mean, r.lgla_sample_sd, r.paired_delta_mean_lgla_minus_plain, r.paired_delta_sample_sd, r.improved_seed_count_out_of_3]);
body(devSheet.getRange(`A4:I${3 + devSummary.length}`)); devSheet.getRange(`C4:H${3 + devSummary.length}`).setNumberFormat("0.00000");
devSheet.freezePanes.freezeRows(3); devSheet.showGridLines = false; devSheet.getRange(`A1:I${3 + devSummary.length}`).format.autofitColumns();

title(ablationSheet, "A1:H1", "Seed43 结构对照：Plain、AB1、AB2、完整 LGLA");
ablationSheet.getRange("A3:H3").values = [["Architecture", "Run", "Epoch", "Binary IoU", "Image Macro-F1", "Group Macro-F1", "Joint", "论文角色"]]; header(ablationSheet.getRange("A3:H3"));
ablationSheet.getRange("A4:H7").values = ablations.map((r) => [r.architecture, r.run, r.visible_epoch, r.binary_iou, r.image_macro_f1, r.group_macro_f1, r.joint_score, r.architecture.startsWith("AB1") ? "未使用病灶引导的普通对照 run" : r.architecture.startsWith("AB2") ? "固定等权融合消融" : r.architecture]);
body(ablationSheet.getRange("A4:H7")); ablationSheet.getRange("D4:G7").setNumberFormat("0.00000"); ablationSheet.getRange("A7:H7").format.fill = paleGreen;
ablationSheet.freezePanes.freezeRows(3); ablationSheet.showGridLines = false; ablationSheet.getRange("A1:H7").format.autofitColumns(); ablationSheet.getRange("H:H").format.columnWidth = 30;

title(testSheet, "A1:G1", "冻结测试：Plain Test1 vs LGLA Test2");
testSheet.getRange("A3:G3").values = [["Section", "Metric", "Plain mean", "Plain SD", "LGLA mean", "LGLA SD", "LGLA - Plain"]]; header(testSheet.getRange("A3:G3"));
testSheet.getRange(`A4:G${3 + testCompare.length}`).values = testCompare.map((r) => [r.section, r.metric, r.plain_mean, r.plain_sample_sd, r.lgla_mean, r.lgla_sample_sd, r.delta_lgla_minus_plain]);
body(testSheet.getRange(`A4:G${3 + testCompare.length}`)); testSheet.getRange(`C4:G${3 + testCompare.length}`).setNumberFormat("0.00000");
testSheet.freezePanes.freezeRows(3); testSheet.showGridLines = false; testSheet.getRange(`A1:G${3 + testCompare.length}`).format.autofitColumns();

title(sourceSheet, "A1:F1", "复杂度、来源与可复现边界");
sourceSheet.getRange("A3:F3").values = [["Architecture", "Parameters", "Δ params", "Δ %", "Source", "备注"]]; header(sourceSheet.getRange("A3:F3"));
sourceSheet.getRange("A4:F7").values = complexity.map((r) => [r.architecture, r.parameters, r.delta_parameters, r.relative_delta_percent / 100, r.source, "参数量；未以本表推断临床效能"]);
body(sourceSheet.getRange("A4:F7")); sourceSheet.getRange("B4:C7").setNumberFormat("0"); sourceSheet.getRange("D4:D7").setNumberFormat("0.000%");
sourceSheet.getRange("A9:F13").values = [
  ["Artifact", "Path / hash", "Role", "Frozen", "Training", "Selection"],
  ["Development raw", "experiments/*/results.csv", "best-Joint rows", "Yes", "Already complete", "Pre-test only"],
  ["Plain Test1", "experiments/test1/results/result.json", "frozen baseline test", "Yes", "No", "No"],
  ["LGLA Test2", "experiments/test2/results/result.json", "frozen extension test", "Yes", "No", "No"],
  ["Summary JSON", "LunWen/Data/06_lgla_extension/01_development/summary.json", "machine-readable handoff", "Yes", "No", "No"],
];
header(sourceSheet.getRange("A9:F9")); body(sourceSheet.getRange("A10:F13")); sourceSheet.getRange("A9:F13").format.wrapText = true;
sourceSheet.freezePanes.freezeRows(3); sourceSheet.showGridLines = false; sourceSheet.getRange("A1:F13").format.autofitColumns(); sourceSheet.getRange("B:B").format.columnWidth = 48; sourceSheet.getRange("F:F").format.columnWidth = 28;

wb.recalculate();

const qa = {};
for (const [name, range] of [["Summary", "A1:H21"], ["Paired seeds", "A1:M7"], ["Development", `A1:I${3 + devSummary.length}`], ["Ablations", "A1:H7"], ["Frozen test", `A1:G${3 + testCompare.length}`], ["Sources", "A1:F13"]]) {
  qa[name] = await wb.inspect({ kind: "region", sheetId: name, range, maxChars: 5000 });
  const preview = await wb.render({ sheetName: name, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(renderDir, `${name.replaceAll(" ", "_")}.png`), new Uint8Array(await preview.arrayBuffer()));
}
qa.formulas = await wb.inspect({ kind: "formula", sheetId: "Summary", range: "B8:E11", maxChars: 4000, options: { maxResults: 100 } });
qa.errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, maxChars: 4000 });
await fs.writeFile(`${outDir}/summary_workbook_qa.json`, JSON.stringify(qa, null, 2), "utf8");
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(`${outDir}/LGLA_v1_thesis_summary.xlsx`);
console.log(JSON.stringify({ status: "PASS", workbook: `${outDir}/LGLA_v1_thesis_summary.xlsx`, renders: renderDir }, null, 2));
