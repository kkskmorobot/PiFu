# Run9 preregistered protocol — 2026-08-01

## Decision basis

Run8 stopped normally at epoch 82 because the preregistered composite score last improved at epoch 62. Run8 produced its first fully feasible checkpoint at epoch 60, but the image-classification branch showed strong confidence overfitting: validation image-classification loss reached its minimum at epoch 32 and later rose above 1.2 while train image-classification loss fell below 0.01.

Run8 proved that within-class lesion-balanced sampling materially improves Run7. Run9 therefore keeps that sampler and tests one conservative classification regularizer.

## Only changed factor

Image-classification cross-entropy uses `label_smoothing=0.05` instead of `0.0`.

The smoothing is applied only to the seven-class image-classification CE. The semantic main and auxiliary losses are unchanged. Run8 architecture, weights initialization, lesion sampler, exact class quotas, epoch length, optimizer, learning-rate schedule, augmentations, image size, batch size, seed, patience, validation split and checkpoint-selection policy remain fixed.

No dropout change, class weight, focal loss, balanced softmax, asymmetric mel-to-nv risk loss, calibration post-processing, new augmentation or test feedback is introduced.

## Hypothesis and acceptance

Primary hypothesis: mild label smoothing prevents the classification head from driving training CE toward zero and improves or preserves validation Macro-F1 without worsening the preregistered mel-to-nv constraint.

The same Run8 acceptance constraints apply:

- foreground mIoU at least 0.518748;
- minority recall at least 0.533498;
- mel-to-nv rate at most 0.1077844311 (18/167);
- primary feasible-checkpoint ranking by validation Macro-F1;
- no internal test inference before final model selection.
