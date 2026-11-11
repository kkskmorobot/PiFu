# Run8 verification — 2026-08-01

## Outcome

Run8 implementation and GPU smoke validation passed. Formal training was launched visibly from PyCharm with the shared `Run8 Lesion Balanced Formal` configuration at 2026-08-01 08:14 (Asia/Hong_Kong).

Formal output:

`results/semantic_multitask_lesion_balanced/skin7_yolo26s_sem_lesion_balanced_seed42`

No test inference was run.

## Controlled-factor audit

- Model: fixed Run7 `Skin7LesionGuidedMultiTaskModel`, 6,616,038 parameters, 17.4 GFLOPs.
- Data split, losses, optimizer, LR schedule, augmentations, batch 16, image size 640, seed 42, 100 epochs, patience 20, and checkpoint selection rules match Run7.
- Epoch length remains 8,512 draws.
- Exact class quotas remain `akiec=800, bcc=1000, bkl=1400, df=320, mel=1400, nv=3200, vasc=392`.
- The only change is uniform lesion selection inside each fixed class quota, followed by uniform image-variant selection inside the lesion.

This class-stratified design is necessary for a clean control. Fully uniform sampling over all 4,042 lesions would change expected NV exposure from 37.6% of images to 64.1% of lesions and would unintentionally remove the established V3 minority-class exposure.

## Metadata and sampler checks

- Train metadata: 8,512 unique image IDs, 4,042 lesions, 2,997 augmented images.
- 1,678 lesions have multiple train images; maximum is 18 images for one lesion.
- Every active full-train image mapped to exactly one metadata lesion and disease.
- `sample_epoch(17)` repeated exactly under seed 42.
- Epoch 17 and epoch 18 sequences differ.
- `set_epoch(17)` reproduced the same epoch-17 sequence and advanced sampler state to 18.
- A sampled full epoch had exactly the seven preregistered class quotas above.

## GPU smoke

Environment: Python 3.13.7, PyTorch 2.8.0+cu129, Ultralytics 8.4.101, NVIDIA GeForce RTX 5080 16 GB.

Command path: `tools/train_skin7_yolo26_lesion_balanced.py --smoke-test`

- 1% train fraction: 85 images, 49 active lesions, 22 batches at batch 4 and image size 320.
- One train epoch completed without loader, forward, backward, or checkpoint-selection errors.
- Full unchanged val split: all 1,504 images completed.
- Smoke runtime recorded in `results.csv`: 23.93 seconds for the epoch/validation record.
- The smoke used no test inference.

The final one-epoch smoke checkpoint is deliberately optimizer-stripped by Ultralytics after training completes and is not a resumable checkpoint. During formal training, the non-final `last.pt` is expected to retain epoch, optimizer, scheduler, scaler, EMA, and train arguments; the Run8 resume script rejects stripped, completed, wrong-name, and wrong-model checkpoints.

## Formal launch verification

- PyCharm run configuration: `.idea/runConfigurations/Run8_Lesion_Balanced_Formal.xml`.
- Resume configuration: `.idea/runConfigurations/Run8_Lesion_Balanced_Resume.xml`.
- Active process was created from the project `.venv` and formal train script.
- Formal loader logged 8,512 images, 4,042 lesions, 1,678 multi-image lesions, maximum 18.
- Formal loader logged the exact baseline class image counts and unique-lesion counts.
- Formal output directory and `args.yaml` were created under experiment 08.

## First formal epoch and resume acceptance

- Epoch 1 completed in 425.275 seconds and the unchanged full val pass completed.
- `results.csv` recorded the first formal row; these warm-up metrics are not used as a Run8 conclusion.
- `last.pt`, `best.pt`, `best_macro_f1.pt`, and `epoch0.pt` were written at 53,310,169 bytes each.
- A read-only `resume_yolo26_lesion_balanced_v8.py --dry-run` accepted the formal `last.pt` as complete.
- The resume check reported completed epoch 1 and deterministic sampler alignment to epoch 2/100.
- The original PyCharm process remained active after the read-only check and continued training.

Ultralytics 8.4.101 does not serialize the EarlyStopping patience counter; that counter restarts after an interruption. Model/EMA, epoch, optimizer, scheduler, scaler, formal run identity, and deterministic Run8 sampler epoch are covered by the resume contract.
