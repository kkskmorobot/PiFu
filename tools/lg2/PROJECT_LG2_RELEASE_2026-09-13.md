# lg2 formal-training release gate

Date: 2026-09-13 (+08:00)

## Release decision

- `PROJECT_LG2_GATE=PASS`
- `LG2_READY_TO_CLICK_GATE=PASS`
- `USER_CONFIRMATION_RECORDED=TRUE`
- `TEST_METRICS_USED=FALSE`
- `EXTERNAL_METRICS_USED=FALSE`
- `FORMAL_TRAINING_USER_CLICK_REQUIRED=TRUE`
- `FORMAL_LONG_TRAINING_STARTED=FALSE`

`USER_CONFIRMATION_RECORDED=TRUE` records the user's explicit instruction in this task to inspect lgx, choose the next experiment, and prepare it up to the PyCharm click. It does not authorize Codex to start formal training. The user must start lg2 manually from the canonical entrypoint.

## Frozen release identity

- Experiment: `lg2_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed42`
- Canonical entrypoint: `tools/lg2/train_lg2_from_pycharm.py`
- Preregistered config SHA-256: `ef019674fe95ae80560285abfc74fe046b72f4b3a473fb05334cae4aef19f170`
- Final lock fingerprint: `16f9505ad452813aac9eb85e1b4cb176d077ff9d16a30ac20965ff3efbd87df4`
- Development manifest SHA-256: `3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`
- Replication identity: exact successful lgx structure at seed42, including classified-batch mean `(g_local-g_global)` squared regularizer, weight `0.01`.

## Gate evidence

- Final lg2 plus retained historical test suite: PASS (`155 passed, 2 subtests passed`).
- CPU exact interrupted/resume comparison: PASS.
- CUDA/AMP exact interrupted/resume comparison: PASS.
- RTX 5080 batch 16, image size 640 capacity/backward/AdamW-step preflight: PASS.
- Final real-development forward/loss-only preparation: PASS; no optimizer, backward, training, test, or external evaluation.
- Formal Trainer constructor-only preflight: PASS; no optimizer and no call to `trainer.train()`.
- Formal lg2 result directory absent at release time.

The full audit is recorded in `tools/lg2/LG2_PREFLIGHT_VERIFICATION_2026-09-13.md`.
