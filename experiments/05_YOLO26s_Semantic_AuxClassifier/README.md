# Experiment 05: YOLO26 semantic + auxiliary image classifier

## Single controlled change

Relative to experiment 04, this run adds one image-level seven-class head. It
globally pools the existing P3 and P4 features, concatenates them, and applies a
small MLP. The semantic head, semantic loss, V3 data, seed, optimizer and data
augmentation remain unchanged. The image head uses ordinary cross-entropy; no
CBAM, proposed attention module, sampler change, focal loss or test feedback is
introduced here.

## Checkpoint selection fixed before training

The validation fitness is fixed in the YAML before results are observed:

`0.50 * Macro-F1 + 0.25 * foreground-mIoU + 0.25 * minority-recall`

An additional penalty applies only when the validation `mel -> nv` rate exceeds
the runs4 reference of `18/167`. All components are logged separately. The
formal report must still use the Pareto view and must not hide a failed
segmentation or safety constraint behind the composite score.

## Start from PyCharm

Run `tools/train_skin7_yolo26_multitask.py`. First use `--dry-run`, then
`--smoke-test`; only after both pass start the default formal run. The script has
no test-evaluation path. Formal output is written beneath this experiment's
`results/` directory and `exist_ok` is false.

## Resume after interruption

Right-click `tools/resume_yolo26_multitask_v5.py` in PyCharm and choose Run.
It validates the formal run's `last.pt`, then restores the completed epoch,
optimizer, LR scheduler, AMP scaler, EMA and global best fitness with the same
custom runs5 trainer and checkpoint-selection rule. Ultralytics 8.4.101 does
not store the EarlyStopping `best_epoch`/patience counter, so patience counting
restarts after resume. Use `--dry-run` to inspect resumability without starting
training. The resume entry has no test path.
