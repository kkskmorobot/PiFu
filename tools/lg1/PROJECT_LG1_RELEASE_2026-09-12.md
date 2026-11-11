# LGLA v1 `lg1` formal-training release gate

Date: 2026-09-12 (+08:00)

## Release decision

- `PROJECT_LG1_GATE=PASS`
- `LG1_READY_TO_CLICK_GATE=PASS`
- `USER_CONFIRMATION_RECORDED=TRUE`
- `TEST_METRICS_USED=FALSE`
- `EXTERNAL_METRICS_USED=FALSE`
- `FORMAL_TRAINING_USER_CLICK_REQUIRED=TRUE`
- `FORMAL_LONG_TRAINING_STARTED=FALSE`

`USER_CONFIRMATION_RECORDED=TRUE` records the user's explicit authorization in this task to prepare the preregistered `lg1` experiment and stop immediately before the PyCharm click. It is not authorization for Codex to start the formal training. The formal run must be started by the user from the canonical PyCharm entrypoint.

## Frozen release identity

- Experiment: `lg1_YOLO26m_Binary8Class_LGLAv1_Seed43`
- Canonical entrypoint: `tools/lg1/train_lg1_from_pycharm.py`
- Preregistered config SHA-256: `d9fde3832b552083e42f1eab744880de507080a29ed7281dda646144f363a782`
- Final lock fingerprint: `ca47d032187061fe8107eb5152d4e3e28026426a6ea7e2df178a5fd146534732`
- Frozen development manifest SHA-256: `3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b`

## Gate evidence

- LGLA unit and historical regression suite: PASS (`81 passed`).
- CPU exact interrupted/resume comparison: PASS.
- CUDA/AMP exact interrupted/resume comparison: PASS.
- RTX 5080, batch 16, image size 640 capacity/backward/optimizer-step preflight: PASS.
- Final real-development forward/loss-only preparation: PASS; no optimizer, backward, training, test, or external evaluation.
- Formal `Trainer` constructor-only preflight: PASS; no optimizer and no call to `trainer.train()`.
- Final formal result directory absent at release time.

The authoritative evidence and the preregistered screening thresholds are summarized in `tools/lg1/LG1_PREFLIGHT_VERIFICATION_2026-09-12.md` and frozen in `tools/lg1/LG1_PROTOCOL_2026-09-12.md`.

