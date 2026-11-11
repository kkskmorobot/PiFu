# Skin7 Dataset V2 Report

## Purpose

`processed_skin7_v2` is the fixed dataset for the data-control experiment in
`runs2` and for all later attention-module experiments. The original
`processed_skin7` dataset and baseline training outputs are unchanged.

## Split safety

- Seed: `42`
- Split unit: `lesion_id`
- Train/val overlap: `0`
- Train/test overlap: `0`
- Val/test overlap: `0`
- V2 val IDs equal V1 val IDs: yes (`1504/1504`)
- V2 test IDs equal V1 test IDs: yes (`1503/1503`)
- Val and test are copied without augmentation.

## V2 distribution

| Split | akiec | bcc | bkl | df | mel | nv | vasc | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1000 | 1200 | 1400 | 400 | 1800 | 2800 | 450 | 9050 |
| val | 49 | 77 | 165 | 18 | 167 | 1006 | 22 | 1504 |
| test | 49 | 77 | 165 | 17 | 167 | 1006 | 22 | 1503 |

## Baseline hard-example mining

The baseline `best.pt` was run only on the 7008 original training images.
Misclassified and low-confidence sources receive a 2x augmentation priority.

Important training-set errors found:

- `mel -> nv`: 57
- `mel -> bkl`: 6
- `akiec -> bkl/mel/nv`: 12 total
- `bcc -> bkl/df`: 2 total
- `bkl -> nv/mel/df/bcc`: 26 total, plus 2 low-confidence
- Difficult NV examples retained during the NV cap: 169

The baseline already fits the original `df` and `vasc` training images, so
their V2 improvement relies on balanced, diverse augmentation rather than
model-error duplication.

## Augmentation policy

- Source selection uses weighted round-robin instead of unrestricted random
  selection.
- Hard sources receive higher sampling priority.
- Paired image/mask transforms include horizontal/vertical flip, rotation
  within +/-20 degrees, scale 0.92-1.08, and translation within +/-4%.
- Photometric changes are deliberately mild: no hue shift, brightness
  multiplier 0.95-1.05, offset +/-5, and saturation multiplier 0.96-1.04.
- A transformed mask must retain 60%-145% of its original foreground area.
- Every generated sample records its source ID, hard-example status, severity,
  random transform parameters, and retained mask area.

## Validation

- Images and labels match for all three splits.
- All 12057 labels are non-empty.
- Every polygon has at least three points and an even coordinate count.
- All normalized coordinates are in `[0, 1]`.
- Training dry-run passed with `train_yolo11s_seg_v2.yaml`.
- Mask audit flags are recorded in `metadata/mask_audit.csv`; they are not
  automatically deleted. This preserves the earlier decision to retain the
  known unusual masks while keeping them visible for later review.

## Reproducibility

Build (refuses to overwrite an existing V2 directory):

```powershell
.venv\Scripts\python.exe prepare_skin7_dataset_v2.py --apply
```

Validate without training:

```powershell
.venv\Scripts\python.exe train_skin7.py --config train_yolo11s_seg_v2.yaml --dry-run
```

The train-only baseline prediction cache is stored at
`Data/skin7_v2_hard_samples.csv` so a rebuild does not repeat GPU mining.

## Experiment mapping

- `runs`: YOLO11s-seg + dataset V1 (existing baseline)
- `runs2`: YOLO11s-seg + dataset V2 (data-control experiment)
- `runs3`: attention control + dataset V2
- `runs4`: proposed improved attention module + dataset V2

Do not use the test split while selecting runs2/runs3/runs4 hyperparameters.
Use validation metrics for selection and evaluate test only after the final
architecture is fixed.
