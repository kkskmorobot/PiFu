import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const stage = "C:/Users/ROBOT/Documents/皮肤识别/.sum1_stage";
const outDir = path.join(stage, "output");
const previewDir = path.join(stage, "previews");
await fs.mkdir(previewDir, { recursive: true });

const readCsv = async (name) => {
  const text = await fs.readFile(path.join(outDir, name), "utf8");
  const lines = text.trim().split(/\r?\n/);
  const parseLine = (line) => {
    const values = [];
    let value = "";
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (quoted && line[i + 1] === '"') { value += '"'; i++; }
        else quoted = !quoted;
      } else if (ch === "," && !quoted) {
        values.push(value); value = "";
      } else value += ch;
    }
    values.push(value);
    return values;
  };
  const headers = parseLine(lines[0]);
  return lines.slice(1).map((line) => Object.fromEntries(headers.map((h, i) => [h, parseLine(line)[i] ?? ""])));
};

const runs = await readCsv("runs.csv");
const families = await readCsv("families.csv");
const complexity = await readCsv("complexity.csv");
const paired = await readCsv("paired.csv");
const val1 = await readCsv("val1.csv");
const sources = JSON.parse(await fs.readFile(path.join(outDir, "sources.json"), "utf8"));

const wb = Workbook.create();
const summary = wb.worksheets.add("Summary");
const runsSheet = wb.worksheets.add("Runs");
const familySheet = wb.worksheets.add("Families");
const complexitySheet = wb.worksheets.add("Complexity");
const pairedSheet = wb.worksheets.add("Paired");
const valSheet = wb.worksheets.add("Val1");
const sourcesSheet = wb.worksheets.add("Sources");
const sheets = [summary, runsSheet, familySheet, complexitySheet, pairedSheet, valSheet, sourcesSheet];
for (const sheet of sheets) sheet.showGridLines = false;
summary.tabColor = "#1F4E78";
valSheet.tabColor = "#C65911";
sourcesSheet.tabColor = "#7F7F7F";

const font = "Arial";
const titleStyle = { font: { name: font, size: 15, bold: true, color: "#1F1F1F" } };
const headerStyle = {
  fill: "#1F4E78",
  font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
};
const bodyFont = { name: font, size: 10, color: "#222222" };
const sectionStyle = { fill: "#D9EAF7", font: { name: font, size: 11, bold: true, color: "#1F1F1F" } };

function setTitle(sheet, title, context) {
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format = titleStyle;
  sheet.getRange("A3:J3").format.borders = { bottom: { style: "thin", color: "#7F7F7F" } };
  sheet.getRange("A4").values = [[context]];
  sheet.getRange("A4:J4").format.font = { name: font, size: 10, italic: true, color: "#595959" };
}

function writeTable(sheet, startRow, headers, rows, tableName) {
  const startCol = 1;
  const headerRange = sheet.getRangeByIndexes(startRow - 1, startCol - 1, 1, headers.length);
  headerRange.values = [headers];
  headerRange.format = headerStyle;
  if (rows.length) {
    const dataRange = sheet.getRangeByIndexes(startRow, startCol - 1, rows.length, headers.length);
    dataRange.values = rows;
    dataRange.format.font = bodyFont;
    dataRange.format.verticalAlignment = "center";
    const fullRange = sheet.getRangeByIndexes(startRow - 1, startCol - 1, rows.length + 1, headers.length);
    const table = sheet.tables.add(fullRange, true, tableName);
    table.style = "TableStyleMedium2";
  }
}

setTitle(summary, "YOLO26m development summary", "Development-only evidence through Val1. No test or external metrics are included.");
summary.getRange("A6:J6").values = [["Locked architecture", "Decision rule", "Seeds", "Feasible checkpoints", "Next stage", "", "", "", "", ""]];
summary.getRange("A6:E6").format = headerStyle;
summary.getRange("A7:E7").values = [["Plain YOLO26m P3/P4", "8-class argmax", "42 / 43 / 44", 0, "Final evaluation preregistration"]];
summary.getRange("A7:E7").format.font = bodyFont;
summary.getRange("D7").format.numberFormat = "0";
summary.getRange("A9:J9").values = [["Three-seed architecture means", "", "", "", "", "", "", "", "", ""]];
summary.getRange("A9:J9").format = sectionStyle;
const summaryHeaders = ["Family", "N", "Binary IoU", "Image F1", "Group F1", "Group recall", "MEL to NV", "Malig. to benign", "Joint", "Feasible"];
writeTable(summary, 10, summaryHeaders, families.map((r) => [
  r.family, Number(r.n), Number(r.binary_iou_mean), Number(r.image_macro_f1_mean), Number(r.group_macro_f1_mean),
  Number(r.group_recall_mean), Number(r.mel_to_nv_mean), Number(r.malignant_to_benign_mean), Number(r.joint_mean), Number(r.feasible_count),
]), "SummaryFamilies");
summary.getRange("B11:J13").format.numberFormat = "0.00000";
summary.getRange("B11:B13").format.numberFormat = "0";
summary.getRange("J11:J13").format.numberFormat = "0";
summary.getRange("A15:J15").values = [["Decision", "", "", "", "", "", "", "", "", ""]];
summary.getRange("A15:J15").format = sectionStyle;
summary.getRange("A16:J18").values = [
  ["Plain has the highest mean Group F1, Group recall and Joint among the three replicated architectures.", "", "", "", "", "", "", "", "", ""],
  ["ECA and ECA+Spatial7 do not provide stable utility gains across seeds.", "", "", "", "", "", "", "", "", ""],
  ["Val1 fixed-cost inference is rejected; risk fell but utility guardrails failed. Formal feasible checkpoint count remains zero.", "", "", "", "", "", "", "", "", ""],
];
summary.getRange("A16:J18").format.font = bodyFont;
summary.getRange("A20:E20").values = [["Model parameters", "", "", "", ""]];
summary.getRange("A20:E20").format = sectionStyle;
writeTable(summary, 21, ["Family", "Parameters", "Delta vs Plain", "Increase", "GFLOPs note"], complexity.map((r) => [
  r.family, Number(r.parameters), Number(r.delta_vs_plain), Number(r.relative_increase_percent) / 100, r.gflops_note,
]), "SummaryComplexity");
summary.getRange("B22:C25").format.numberFormat = "#,##0";
summary.getRange("D22:D25").format.numberFormat = "0.0000%";

const utilityChart = summary.charts.add("bar", [summary.getRange("A10:A13"), summary.getRange("E10:E13"), summary.getRange("I10:I13")]);
utilityChart.title = "Mean Group F1 and Joint";
utilityChart.titleTextStyle.typeface = font;
utilityChart.titleTextStyle.fontSize = 12;
utilityChart.legend = { position: "top", textStyle: { typeface: font, fontSize: 10 } };
utilityChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 10 } };
utilityChart.yAxis = { numberFormatCode: "0.00", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 10 } };
utilityChart.setPosition("L3", "T16");
const riskChart = summary.charts.add("bar", [summary.getRange("A10:A13"), summary.getRange("G10:G13"), summary.getRange("H10:H13")]);
riskChart.title = "Mean group risk rates";
riskChart.titleTextStyle.typeface = font;
riskChart.titleTextStyle.fontSize = 12;
riskChart.legend = { position: "top", textStyle: { typeface: font, fontSize: 10 } };
riskChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 10 } };
riskChart.yAxis = { numberFormatCode: "0.00", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 10 } };
riskChart.setPosition("L18", "T31");

setTitle(runsSheet, "Best-Joint run records", "One row per completed YOLO26m experiment. Metrics come from each run's maximum development Joint row.");
const runHeaders = ["Run", "Seed", "Family", "Role", "Best epoch", "Epochs", "Binary IoU", "Image F1", "Group F1", "Group recall", "MEL to NV", "Malig. to benign", "NLL", "Brier", "ECE", "Joint", "Risk feasible"];
writeTable(runsSheet, 6, runHeaders, runs.map((r) => [
  Number(r.run), Number(r.seed), r.family, r.role, Number(r.best_epoch), Number(r.epochs_recorded), Number(r.binary_iou), Number(r.image_macro_f1),
  Number(r.group_macro_f1), Number(r.group_recall), Number(r.mel_to_nv), Number(r.malignant_to_benign), Number(r.nll), Number(r.brier), Number(r.ece), Number(r.joint), r.risk_feasible === "True",
]), "RunsTable");
runsSheet.getRange(`A7:B${6 + runs.length}`).format.numberFormat = "0";
runsSheet.getRange(`E7:F${6 + runs.length}`).format.numberFormat = "0";
runsSheet.getRange(`G7:P${6 + runs.length}`).format.numberFormat = "0.00000";
runsSheet.freezePanes.freezeRows(6);

setTitle(familySheet, "Three-seed family statistics", "Means and sample standard deviations for Plain, ECA and ECA+Spatial7.");
const familyHeaders = Object.keys(families[0]);
writeTable(familySheet, 6, familyHeaders, families.map((r) => familyHeaders.map((h) => h === "family" ? r[h] : Number(r[h]))), "FamiliesTable");
familySheet.getRange(`B7:${String.fromCharCode(64 + familyHeaders.length)}9`).format.numberFormat = "0.00000";
familySheet.getRange("B7:B9").format.numberFormat = "0";
familySheet.getRange(`${String.fromCharCode(64 + familyHeaders.length)}7:${String.fromCharCode(64 + familyHeaders.length)}9`).format.numberFormat = "0";

setTitle(complexitySheet, "Model complexity", "Parameter counts come from the corresponding frozen preflight reports.");
const complexityHeaders = ["Family", "Source run", "Parameters", "Delta vs Plain", "Increase", "GFLOPs note"];
writeTable(complexitySheet, 6, complexityHeaders, complexity.map((r) => [
  r.family, Number(r.source_run), Number(r.parameters), Number(r.delta_vs_plain), Number(r.relative_increase_percent) / 100, r.gflops_note,
]), "ComplexityTable");
complexitySheet.getRange(`B7:D${6 + complexity.length}`).format.numberFormat = "#,##0";
complexitySheet.getRange(`E7:E${6 + complexity.length}`).format.numberFormat = "0.0000%";

setTitle(pairedSheet, "Paired architecture deltas", "Candidate minus Plain at the same seed; mean rows summarize the three paired differences.");
const pairedHeaders = Object.keys(paired[0]);
writeTable(pairedSheet, 6, pairedHeaders, paired.map((r) => pairedHeaders.map((h) => h === "comparison" || h === "seed" ? r[h] : Number(r[h]))), "PairedTable");
pairedSheet.getRange(`C7:${String.fromCharCode(64 + pairedHeaders.length)}${6 + paired.length}`).format.numberFormat = "+0.00000;-0.00000;0.00000";

setTitle(valSheet, "Val1 fixed-cost decision", "Frozen-model development comparison. The fixed-cost rule failed the preregistered utility guardrails.");
const valHeaders = Object.keys(val1[0]);
writeTable(valSheet, 6, valHeaders, val1.map((r) => valHeaders.map((h) => ["seed", "source_run"].includes(h) ? Number(r[h]) : Number(r[h]))), "Val1Table");
valSheet.getRange(`A7:B${6 + val1.length}`).format.numberFormat = "0";
valSheet.getRange(`C7:${String.fromCharCode(64 + valHeaders.length)}${6 + val1.length}`).format.numberFormat = "0.00000";
valSheet.getRange(`A${9 + val1.length}:H${9 + val1.length}`).values = [["Acceptance gate", "FAIL", "Decision", "Retain argmax", "Training", false, "Test/external", false]];
valSheet.getRange(`A${9 + val1.length}:H${9 + val1.length}`).format.font = { name: font, size: 10, bold: true, color: "#9C0006" };

setTitle(sourcesSheet, "Source audit", "Hashes identify every source file used by this development-only summary.");
writeTable(sourcesSheet, 6, ["Kind", "Run", "Rows", "SHA-256", "Path"], sources.map((r) => [r.kind, String(r.run), Number(r.rows), r.sha256, r.path]), "SourcesTable");
sourcesSheet.getRange(`C7:C${6 + sources.length}`).format.numberFormat = "0";
sourcesSheet.freezePanes.freezeRows(6);

for (const sheet of sheets) {
  const used = sheet.getUsedRange();
  used.format.verticalAlignment = "center";
  used.format.autofitColumns();
  used.format.autofitRows();
}
summary.getRange("A:A").format.columnWidth = 28;
summary.getRange("B:J").format.columnWidth = 15;
runsSheet.getRange("C:C").format.columnWidth = 24;
runsSheet.getRange("D:D").format.columnWidth = 18;
sourcesSheet.getRange("D:D").format.columnWidth = 68;
sourcesSheet.getRange("E:E").format.columnWidth = 120;
sourcesSheet.getRange(`D7:E${6 + sources.length}`).format.wrapText = false;

wb.recalculate();

const summaryInspect = await wb.inspect({ kind: "table", range: "Summary!A2:J18", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 12 });
console.log(summaryInspect.ndjson);
const errorInspect = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });
console.log(errorInspect.ndjson);
const drawings = await wb.inspect({ kind: "drawing", sheetId: "Summary", maxChars: 4000 });
console.log(drawings.ndjson);

for (const sheet of sheets) {
  const preview = await wb.render({ sheetName: sheet.name, autoCrop: "all", scale: sheet.name === "Sources" ? 0.7 : 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheet.name}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(outDir, "summary.xlsx"));
console.log("SUM1_WORKBOOK=PASS");
