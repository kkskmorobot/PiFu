# lgx formal-training release gate

Date: 2026-09-13 (+08:00)

## Release decision

- `PROJECT_LGX_GATE=PASS`
- `LGX_READY_TO_CLICK_GATE=PASS`
- `USER_CONFIRMATION_RECORDED=TRUE`
- `TEST_METRICS_USED=FALSE`
- `EXTERNAL_METRICS_USED=FALSE`
- `FORMAL_TRAINING_USER_CLICK_REQUIRED=TRUE`
- `FORMAL_LONG_TRAINING_STARTED=FALSE`

`USER_CONFIRMATION_RECORDED=TRUE` records the user's explicit instruction in this task to inspect lg1, choose the next experiment, and prepare it up to the PyCharm click. It does not authorize Codex to start formal training. The user must start lgx manually from the canonical entrypoint.

## Frozen release identity

- Experiment: `lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43`
- Canonical entrypoint: `tools/lgx/train_lgx_from_pycharm.py`
- Preregistered config SHA-256: `15a854ceee23149017dab986d18b6729359eabcc58be132990efe1e9df0f60df`
- Final lock fingerprint: `3a9aa4156828bd2e5af1b1f44a62929041cde5895d01ed995c8c5af823186ac0`
- Development manifest SHA-256: `3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`
- Single revision: classified-batch mean `(g_local-g_global)` squared regularizer, weight `0.01`.

## Gate evidence

- Final lgx plus retained historical test suite: PASS (`88 passed`).
- CPU exact interrupted/resume comparison: PASS.
- CUDA/AMP exact interrupted/resume comparison: PASS.
- RTX 5080 batch 16, image size 640 capacity/backward/AdamW-step preflight: PASS.
- Final real-development forward/loss-only preparation: PASS; no optimizer, backward, training, test, or external evaluation.
- Formal Trainer constructor-only preflight: PASS; no optimizer and no call to `trainer.train()`.
- Formal lgx result directory absent at release time.

The full audit is recorded in `tools/lgx/LGX_PREFLIGHT_VERIFICATION_2026-09-13.md`.

