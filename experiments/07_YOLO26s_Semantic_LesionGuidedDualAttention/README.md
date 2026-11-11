# Experiment 07: lesion-guided multi-scale dual attention

## Controlled change

Runs7 returns to the runs5 base and changes only the image-classification
feature aggregation. The unchanged semantic P3 head produces an eight-class
probability map. `1 - P(background)` becomes a soft lesion mask that guides
P3/8, P4/16 and P5/32 pooling. Each scale also receives an identity-initialized
ECA-style channel gate before pooling; the three descriptors are concatenated
for the seven-class image classifier.

The semantic mask is detached in the classifier branch. Therefore this run
does not add a consistency loss or a second gradient path into the semantic
classifier. Loss and lesion-balanced sampling remain reserved for runs8.

## Preregistered implementation

- P3/P4/P5 sources: YOLO26-sem layers 16, 13 and 10.
- Spatial attention: bilinearly resized soft foreground probability.
- Pooling: 50% global average plus 50% normalized mask-guided average.
- Channel attention: ECA-style kernel 5, initialized as an identity gate.
- P5 classifier weights start at zero contribution; existing P3/P4 classifier
  initialization is preserved.
- Semantic main/auxiliary paths see the original P3/P4 features and are not
  modified by the proposed module.

## Fixed protocol

- Same V3 data and `models/yolo26s-sem.pt` initialization as runs5/runs6.
- seed=42, deterministic=true, batch=16, imgsz=640.
- AdamW, lr0=0.001, cosine schedule, 100 epochs, patience=20.
- Same augmentation, semantic losses and image CE weight 0.5 as runs5.
- No CBAM, focal/balanced loss, sampler change, extra augmentation or test use.
- Same online composite fitness and formal feasibility constraints as runs6.
- `best_macro_f1.pt` tracks the primary image metric.
- `best_feasible_macro_f1.pt` exists only if foreground mIoU >= 0.518748,
  minority Recall >= 0.533498 and mel-to-nv rate <= 18/167.

## Launch

Run `tools/train_skin7_yolo26_lesion_guided.py --dry-run`, then
`--smoke-test`. Start formal training by running the same file without
arguments from PyCharm. Use `tools/resume_yolo26_lesion_guided_v7.py` only
after a genuine interruption with a complete `last.pt`.
