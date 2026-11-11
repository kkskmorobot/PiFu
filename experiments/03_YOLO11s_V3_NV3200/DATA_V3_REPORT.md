# Skin7 Dataset V3 Report

## Decision

V3 is a controlled data ablation created after the broader V2 balancing and
augmentation policy reduced held-out test performance. It changes exactly one
training-data variable from the original V1 baseline: real NV images are
reduced from 3500 to 3200.

## Locked controls

- Split seed: `42`
- Split unit: `lesion_id`
- V3 validation set equals V1 validation set: yes (`1504/1504`)
- V3 test set equals V1 test set: yes (`1503/1503`)
- All non-NV V3 training metadata equals V1: yes
- Offline augmentation implementation equals V1: yes
- Online training augmentation and all optimizer settings equal V1: yes
- Hard-example weighting: disabled
- Additional V2 rotation/scale/translation policy: disabled

## Distribution

| Split | akiec | bcc | bkl | df | mel | nv | vasc | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 800 | 1000 | 1400 | 320 | 1400 | 3200 | 392 | 8512 |
| val | 49 | 77 | 165 | 18 | 167 | 1006 | 22 | 1504 |
| test | 49 | 77 | 165 | 17 | 167 | 1006 | 22 | 1503 |

Compared with V1, exactly 300 training NV images are removed. V3 NV is a
lesion-safe subset of V1 NV. No validation or test sample is changed.

## Validation

- Image/label stem matching passed for train, val, and test.
- Every label is non-empty and uses the expected seven-class mapping.
- Every polygon has at least three points, an even coordinate count, and
  normalized coordinates in `[0, 1]`.
- `train_skin7.py --config train_yolo11s_seg_v3.yaml --dry-run` passed.
- Machine-readable audit: `Data/processed_skin7_v3/metadata/v3_audit.json`.

## Experiment mapping

- `runs`: original YOLO11s-seg + V1 baseline
- `runs2`: broad V2 balancing/augmentation experiment
- `runs3`: YOLO11s-seg + controlled V3 NV3200 experiment
- `runs4`: attention control on the selected stable dataset
- `runs5`: proposed improved attention module

The V3 test split must remain untouched during training. Select the model with
validation metrics and use `best.pt` for the final held-out evaluation.
