# lg3 formal-training release gate

Date: 2026-09-14 (+08:00)

## Release decision

- `PROJECT_LG3_GATE=PASS`
- `LG3_READY_TO_CLICK_GATE=PASS`
- `USER_CONFIRMATION_RECORDED=TRUE`
- `TEST_METRICS_USED=FALSE`
- `EXTERNAL_METRICS_USED=FALSE`
- `FORMAL_TRAINING_USER_CLICK_REQUIRED=TRUE`
- `FORMAL_LONG_TRAINING_STARTED=FALSE`

`USER_CONFIRMATION_RECORDED=TRUE` records the user's explicit instruction in this task to inspect lg2, choose the next experiment, and prepare it up to the PyCharm click. It does not authorize Codex to start formal training. The user must start lg3 manually from the canonical entrypoint.

## Frozen release identity

- Experiment: `lg3_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed44`
- Canonical entrypoint: `tools/lg3/train_lg3_from_pycharm.py`
- Preregistered config SHA-256: `cac7a51ea7a5ee7780748cf860005d03976a3333336a149570504709e8ec90c9`
- Final lock fingerprint: `4a7c29f336b7ea28d8fbfe38e5e51cdff21f68ba418c78fddedbceb6956642aa`
- Development manifest SHA-256: `3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`
- Replication identity: exact successful lgx structure at seed44, including classified-batch mean `(g_local-g_global)` squared regularizer, weight `0.01`.

## Gate evidence

- Final lg3 plus retained historical test suite: PASS (`162 passed, 2 subtests passed`).
- CPU exact interrupted/resume comparison: PASS.
- CUDA/AMP exact interrupted/resume comparison: PASS.
- RTX 5080 batch 16, image size 640 capacity/backward/AdamW-step preflight: PASS.
- Final real-development forward/loss-only preparation: PASS; no optimizer, backward, training, test, or external evaluation.
- Formal Trainer constructor-only preflight: PASS; no optimizer and no call to `trainer.train()`.
- Formal lg3 result directory absent at release time.

The full audit is recorded in `tools/lg3/LG3_PREFLIGHT_VERIFICATION_2026-09-14.md`.
