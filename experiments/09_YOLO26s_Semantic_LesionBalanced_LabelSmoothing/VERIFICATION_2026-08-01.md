# Run9 verification — 2026-08-01

## Controlled-factor audit

- Fixed parent: Run8 within-class lesion-balanced sampler and Run7 lesion-guided model.
- Only changed objective term: seven-class image CE uses `label_smoothing=0.05`.
- Semantic main/auxiliary losses, image CE weight 0.5, model initialization, 6,616,038 parameters, optimizer, LR schedule, augmentations, exact class quotas, 8,512 draws per epoch, batch 16, image size 640, seed 42, patience 20 and validation selection constraints are unchanged.
- No dropout, class-weight, focal, balanced-softmax, asymmetric-risk, calibration or augmentation change is included.
- No test inference was run.

## Static and mathematical checks

- Python 3.13 compilation passed for the model/loss, train entry and resume entry.
- `Skin7LabelSmoothedMultiTaskModel` inherits the fixed Run7 model.
- `Skin7LabelSmoothedTrainer` inherits the Run8 lesion-balanced trainer.
- `label_smoothing=0.0` exactly reproduced ordinary PyTorch CE in a tensor check.
- For an overconfident correct example, epsilon 0.05 increased CE from 0.0000621 to 0.350062 and retained a finite nonzero gradient, confirming the intended confidence penalty.
- Dry-run validated the V3 train/val/test structure but did not infer test.

## RTX 5080 smoke

Environment: Python 3.13.7, PyTorch 2.8.0+cu129, Ultralytics 8.4.101, NVIDIA GeForce RTX 5080 16 GB.

- 1% train fraction: 85 images, 49 active lesions, 22 batches at batch 4 and image size 320.
- Run8 sampler audit remained active with fixed class quotas.
- One forward/backward train epoch completed.
- The unchanged full val split of 1,504 images completed.
- Model remained 6,616,038 parameters and 17.4 GFLOPs before fusion.
- Smoke row time: 22.7055 seconds; train image CE 1.88191; val image CE 1.50062.
- Output: `smoke/label_smoothing_005_1pct_full_val_smoke`.

The one-epoch smoke checkpoint is finalized and optimizer-stripped by Ultralytics, so formal resume acceptance will be checked against the first non-final formal `last.pt` after PyCharm launch.

## Formal launch

- Launched visibly from PyCharm using `Run9 Label Smoothing Formal` on 2026-08-01.
- Formal output: `results/semantic_multitask_lesion_balanced_label_smoothing/skin7_yolo26s_sem_lesion_balanced_ls005_seed42`.
- Full loader audit logged 8,512 draws, 4,042 lesions, 1,678 multi-image lesions and maximum 18 images per lesion.
- Exact baseline class quotas and unique-lesion class counts were logged unchanged.
- The formal criterion logged: semantic loss unchanged and image classification CE `label_smoothing=0.05`.
- Epoch 1 entered training at batch 16 and image size 640 on the RTX 5080.

## First formal epoch and safe resume audit

- Epoch 1 completed in 416.131 seconds and the full unchanged val pass completed.
- Train image CE was 1.26768; val image CE was 1.68440. The warm-up metrics are not used as a Run9 conclusion.
- `last.pt`, `best.pt`, `best_macro_f1.pt`, and `epoch0.pt` were written at 53,310,297 bytes each.
- The original PyCharm process continued after checkpoint creation.
- The resume `--dry-run` was hardened to inspect ZIP members and pickle opcodes without deserializing the pickle payload.
- Safe audit found structural epoch 0, 953 archive members, 947 tensor-storage members, a nonempty optimizer structure, the formal Run9 name, and `skin7_label_smoothed.Skin7LabelSmoothedMultiTaskModel`.
- No second training process was started and test was not evaluated.

Actual resume intentionally deserializes only the project-generated trusted checkpoint because restoring model/EMA and optimizer tensors requires it. The safe dry-run provides non-executing structural acceptance before that action.
