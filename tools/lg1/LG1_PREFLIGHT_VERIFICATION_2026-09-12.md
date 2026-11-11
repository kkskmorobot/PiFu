# `lg1` LGLA v1 preflight verification

Date: 2026-09-12 (+08:00)

## Final decision

`lg1` is ready for the user's PyCharm click. All authoritative preflights pass. No formal long training was started, no formal result directory was created, and no test, ISIC2018 external, or PAD-UFES-20 data or metrics were used.

## Preregistration and frozen comparison

- Base comparison: Run22 plain seed-43.
- Run22 development reference: Binary IoU `0.76377`; Image Macro-F1 `0.55017`; Group Macro-F1 `0.62811`; MEL-to-NV `0.40816`; Joint Score `0.69263`.
- `lg1` screening PASS gate: Group Macro-F1 delta at least `+0.005`; Joint Score delta at least `+0.003`; Image Macro-F1 delta at least `0`; Binary IoU delta at least `-0.005`.
- Diagnostic gate: local-gate mean in `[0.05, 0.95]`, local-gate standard deviation at least `0.001`, all gate and attention statistics finite.
- Config SHA-256: `d9fde3832b552083e42f1eab744880de507080a29ed7281dda646144f363a782`.
- Final lock fingerprint: `ca47d032187061fe8107eb5152d4e3e28026426a6ea7e2df178a5fd146534732`.

## Architecture checks

- Classification-only LGLA on P3/P4; the semantic head and its inputs remain unchanged.
- Predicted semantic attention is detached before use by the classifier.
- Ground-truth masks are never classifier inputs.
- Global average and lesion-weighted local descriptors use shared per-scale 128-dimensional projections and a learned two-way softmax gate.
- Initial gate is exactly 0.5/0.5; empty-attention fallback is global average.
- P5, ROIAlign, GeM, risk loss, loss reweighting, label smoothing, new sampling, and new augmentation are absent.

## Authoritative PASS evidence

1. Unit and regression tests: `81 passed in 2.46s` across `lg1` and the Run18 through Run33 retained tests.
2. CPU checkpoint/resume: `preflight/lg1_checkpoint_resume_v1/20260912_210900/PREFLIGHT_REPORT.json`; exact live model, EMA, optimizer, RNG, and data-order equivalence at absolute tolerance `0`.
3. CUDA/AMP checkpoint/resume: `preflight/lg1_checkpoint_resume_v1/20260912_211026/PREFLIGHT_REPORT.json`; exact live model, EMA, optimizer, RNG, and data-order equivalence, maximum difference `0`.
4. RTX 5080 capacity: `preflight/lg1_yolo26m_capacity_v1/batch16_img640_amp_20260912_v1.json`; batch `16`, image size `640`, AMP, full loss backward and AdamW step PASS. Peak allocated memory `6,460,668,416` bytes (`40.515%` of GPU capacity). Classification-only backward produced no semantic-head gradients.
5. Final preparation: `tools/lg1/LG1_PREFLIGHT_REPORT.json`; real development-role batch forward/loss PASS, parameters `14,540,748`, formal training not started, formal result directory not created, forbidden-data flags all false.
6. Ready marker: `tools/lg1/LG1_READY_TO_TRAIN.json`; status PASS and `ready_to_click_pycharm=true`.
7. Formal constructor: `preflight/lg1_formal_trainer_constructor_v1/lgla_v1_constructor_20260912_v1/CONSTRUCTOR_REPORT.json`; formal parameters and LGLA classes PASS, optimizer not created, `trainer.train()` not called.

An earlier CUDA/AMP probe at `20260912_210927` detected a mixed-precision dtype mismatch in the classification projections. The implementation was corrected to accumulate pooling in FP32 and cast descriptors to the projection dtype. The authoritative post-fix CUDA/AMP run at `20260912_211026`, the capacity preflight, the final preparation, and the formal constructor preflight all pass.

## Development-data boundary

- Train: total `23,462`; segmentation `10,483`; classification `13,715`; both `736`.
- Validation: total `1,744`; segmentation `292`; classification `1,488`; both `36`.
- Development manifest is byte-identical to the frozen Run22 development manifest.
- Test key is null; external candidates are absent.
- `imaplusplus_test=false`, `isic2018_external=false`, `pad_ufes_20=false`.

## Stop point

The canonical PyCharm entrypoint is `tools/lg1/train_lg1_from_pycharm.py`. At verification time, the formal result directory `experiments/lg1_YOLO26m_Binary8Class_LGLAv1_Seed43/results/lg1_yolo26m_lgla_v1_seed43` did not exist. The next permitted action is the user's manual PyCharm run of the canonical entrypoint with no arguments.
