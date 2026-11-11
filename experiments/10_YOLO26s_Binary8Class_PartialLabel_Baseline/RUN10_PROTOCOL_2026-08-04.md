# Run10 preregistered protocol — 2026-08-04

## Role

Run10 is the first formal baseline under the accepted binary lesion segmentation
plus official eight-class partial-label contract. It is not presented as a
causal one-factor comparison with Run9 because the data target, taxonomy and
model contract form one inseparable generation change.

The single registered transition is
`binary_segmentation_plus_official_8class_partial_labels_v1`. Within that
contract Run10 adds no experimental attention module, lesion sampler, class
weight, focal/asymmetric loss, label smoothing, calibration fitting or external
feedback. Every later run must change exactly one factor from this baseline.

## Frozen data

- Development train: 23,462 images; segmentation supervision 10,483;
  classification supervision 13,715; both 736.
- Development val: 1,744 images; segmentation supervision 292;
  classification supervision 1,488; both 36.
- The experiment-local development manifest is a byte-stable train/val-only
  projection of the accepted combined manifest. It does not contain the 634
  IMA++ test rows.
- `test: null`; PAD-UFES-20 and ISIC2018 external candidates are absent from the
  training data YAML and manifest.
- No accepted source manifest, split, mask value or test rule is modified.

## Frozen model and optimization

- Initialization: `models/yolo26s-sem.pt`; pretrained binary semantic weights
  are transferred, while the plain P3/P4 global-pooling eight-logit image head
  is initialized under seed 42.
- Image-classification loss weight: 0.5; no label smoothing.
- Standard seeded epoch shuffle; every development-train sample occurs once per
  epoch. No lesion-balanced or class-balanced sampler.
- 100 epochs, patience 20, batch 16, image size 640, AdamW, lr0 0.001,
  cosine lrf 0.01, weight decay 0.0005, warmup 3 epochs, AMP on, workers 8.
- Mosaic, MixUp, CutMix and CopyPaste are exactly zero. Other augmentations are
  frozen in `run10_preregistered_config_v1.yaml`; implicit classification-only
  AutoAugment and random erasing defaults are explicitly disabled.

## Checkpoint and interruption

- Full bundle v2 is published only at completed-epoch boundaries with
  `_SUCCESS`, file hashes and a configuration/data/code fingerprint.
- Bundle contents include live FP32 model, EMA, optimizer, AMP scaler,
  scheduler, early stopper, selection state, Python/NumPy/Torch CPU/CUDA RNG
  and data progress.
- `resume=auto` selects only the newest complete compatible bundle; an explicit
  bundle path is also supported. Incomplete or fingerprint-mismatched bundles
  are rejected.
- SIGINT/SIGTERM requests a stop at the next completed epoch. An exception
  before that boundary is recorded as incomplete and is not auto-resumable.

## Frozen val selection

- `joint_score=sqrt(binary_iou * group_macro_f1)`.
- Feasible iff group mel->nv rate is at most 0.1077844311 and mel support is
  present.
- Primary checkpoint alias: `best_feasible_joint`.
- Tie order: higher group Macro-F1, then higher binary IoU, then earlier epoch.
- If no feasible checkpoint exists, Run10 is a constraint failure. The best
  unconstrained joint checkpoint may be reported diagnostically but cannot be
  promoted using test or external results.
- Semantic threshold is 0.5; classification is argmax; temperature is 1.0.
  No temperature or threshold is fitted on external data.

## Protected evaluation boundary

Training and model selection use development val only. IMA++ internal test,
ISIC2018 external segmentation and PAD-UFES-20 external classification remain
locked until model/config/checkpoint/evaluator hashes are frozen and the user
gives a new explicit authorization. Results cannot flow back into Run10.
