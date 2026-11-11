# lgx gate-balanced LGLA v1 preflight verification

Date: 2026-09-13 (+08:00)

## Final decision

lgx is ready for the user's PyCharm click. All authoritative preflights pass. Formal training has not started, the formal result directory does not exist, and no test, external, or PAD-UFES-20 data or metrics were used.

## Frozen single-factor revision

- Parent lg1 failure: local gate mean `8.5977e-13`, population std `3.4094e-12`; the local branch was effectively disabled.
- Only lgx change: add `0.01 * (mean_classified(g_local-g_global))^2` inside the existing image-classification loss weight and batch-size scaling.
- The regularizer uses only samples with official classification labels and permits per-sample gate variation.
- Inference architecture, P3/P4 attention, detach, pooling, classifier, seed, optimizer, augmentation, early stopping, validation and checkpoint selection remain unchanged.
- Config SHA-256: `15a854ceee23149017dab986d18b6729359eabcc58be132990efe1e9df0f60df`.
- Final lock fingerprint: `3a9aa4156828bd2e5af1b1f44a62929041cde5895d01ed995c8c5af823186ac0`.

## Authoritative PASS evidence

1. Unit and historical regression tests: `88 passed in 2.59s`.
2. CPU checkpoint/resume: `preflight/lgx_checkpoint_resume_v1/20260913_035325/PREFLIGHT_REPORT.json`; live model, EMA, optimizer, RNG and second-epoch order are exactly equal at tolerance `0`.
3. CUDA/AMP checkpoint/resume: `preflight/lgx_checkpoint_resume_v1/20260913_035355/PREFLIGHT_REPORT.json`; all compared state is exactly equal, maximum difference `0` under tolerance `1e-6`.
4. RTX 5080 capacity: `preflight/lgx_yolo26m_capacity_v1/batch16_img640_amp_20260913_v1.json`; batch `16`, image size `640`, AMP, five finite loss items, full backward and AdamW step PASS. Peak reserved memory is `6,914,310,144` bytes (`40.515%` of device capacity); classification-only backward does not reach either semantic classifier.
5. Final preparation: `tools/lgx/LGX_PREFLIGHT_REPORT.json`; real development batch forward/loss PASS, parameters `14,540,748`, formal training not started, formal result directory not created, forbidden-data flags false.
6. Ready marker: `tools/lgx/LGX_READY_TO_TRAIN.json`; status PASS and `ready_to_click_pycharm=true`.
7. Formal constructor: `preflight/lgx_formal_trainer_constructor_v1/lgla_v1_gate_balance_constructor_20260913_v1/CONSTRUCTOR_REPORT.json`; formal parameters, gate regularizer and class construction PASS; optimizer not created and `trainer.train()` not called.

The first CPU attempt at `20260913_035251` correctly stopped because validation also computes loss while the model is in eval mode and the transient gate logits had only been retained during train mode. The implementation was corrected to retain the most recent transient logits in both modes without changing public inference outputs or checkpoint state. The authoritative CPU/GPU resume runs, capacity test, final preparation, constructor preflight and final 88-test suite all pass after that correction.

## Development-data boundary

- Train: total `23,462`; segmentation `10,483`; classification `13,715`; both `736`.
- Validation: total `1,744`; segmentation `292`; classification `1,488`; both `36`.
- The development manifest is byte-identical to frozen Run22.
- YAML test key is null; external candidates are absent.
- `imaplusplus_test=false`, `isic2018_external=false`, `pad_ufes_20=false`.

## Stop point

The canonical PyCharm entrypoint is `tools/lgx/train_lgx_from_pycharm.py`. At verification time, `experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43/results/lgx_yolo26m_lgla_v1_gate_balance_seed43` did not exist. The next permitted action is the user's manual PyCharm run of the canonical entrypoint with no arguments.

