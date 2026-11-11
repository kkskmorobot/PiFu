# Run8 preregistered protocol — 2026-08-01

## Decision basis

Run7 stopped normally after epoch 55. It did not produce a checkpoint satisfying all preregistered constraints, but lesion-level F1 improved over its image-level behavior. Training image-classification loss continued falling while validation loss worsened, consistent with repeated image variants over-weighting some lesions.

The V3 train metadata contains 8,512 images from 4,042 lesions. A total of 1,678 lesions have more than one active image, and the largest lesion contributes 18 images.

## Only changed factor

Run8 keeps the complete Run7 architecture, losses, optimizer, learning-rate schedule, augmentations, batch size, image size, seed, epoch count, patience, validation data, and checkpoint selection rules.

Only the train sampler changes:

1. Preserve the existing per-epoch image quota of every class exactly: akiec 800, bcc 1,000, bkl 1,400, df 320, mel 1,400, nv 3,200, and vasc 392.
2. Within each class, select one of its active training lesions uniformly with replacement.
3. Select one active image belonging to that lesion uniformly, then shuffle all class draws together.
4. Draw exactly 8,512 samples per epoch, preserving the previous epoch length and class exposure.

This is within-class lesion-balanced sampling, not a new class-balancing policy. The class mix is identical to the baseline image counts; no focal loss, balanced softmax, risk loss, new augmentation, or test feedback is introduced.

## Acceptance and safety

- Sampler metadata must map every active train image to exactly one lesion and class.
- Sampler order must be deterministic under `seed + logical_epoch`.
- Resume must restore the full Ultralytics state and align sampling to `start_epoch`.
- Validation stays unchanged and the internal test split is not inferred during selection.
- Formal output is isolated under `experiments/08_YOLO26s_Semantic_LesionBalancedSampler`.
- Pre-registered selection constraints and score weights are identical to Run7.
