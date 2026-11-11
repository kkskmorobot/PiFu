# Experiment 06: runs5 + one standard CBAM

## Controlled change

Runs6 adds exactly one standard CBAM to the existing P3/8 neck-fusion feature
before that feature enters the semantic head and the P3/P4 auxiliary image
classifier. CBAM applies shared avg/max-pool channel attention followed by a
7x7 spatial-attention convolution. Reduction ratio is 16.

No other model, data or optimization variable changes relative to runs5. The
model starts from the same `models/yolo26s-sem.pt`; it does not fine-tune the
runs5 `best.pt`.

## Fixed protocol

- V3 data; seed=42; batch=16; imgsz=640.
- AdamW, lr0=0.001, cosine schedule, 100 epochs, patience=20.
- Same augmentation, semantic losses and image CE weight 0.5 as runs5.
- No focal loss, sampler change, proposed lesion-guided attention or test use.
- Periodic checkpoints every five epochs.
- `best_macro_f1.pt` tracks the primary image metric.
- `best_feasible_macro_f1.pt` is created only when foreground mIoU >= 0.518748,
  minority Recall >= 0.533498 and mel-to-nv rate <= 18/167.
- `best.pt` remains the same online composite fitness used in runs5, so the
  historical control remains directly comparable.

## Launch

Run `tools/train_skin7_yolo26_cbam.py --dry-run`, then `--smoke-test`. Formal
training is started by running the same file without arguments from PyCharm.
Use `tools/resume_yolo26_cbam_v6.py` only after a genuine interruption.

