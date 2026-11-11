# Experiment 05 implementation verification

Date: 2026-07-20

## Checks completed

- Python syntax compilation passed for the runs4 fix, runs5 model/trainer and unified evaluator.
- Runs4 and runs5 dry-runs passed with V3 counts 8512/1504/1503.
- The runs4 standalone path explicitly resolved the 8-class evaluation YAML with `background=7`.
- The runs5 in-memory forward/loss test passed for semantic main, semantic auxiliary and image-classification outputs.
- A one-epoch, 1% runs5 smoke-test completed on CUDA and validated all 1504 val images.
- No test inference was executed.

The smoke-test initially exposed that `SemanticDataset` does not preserve instance `cls/batch_idx`, and that validation disables the semantic auxiliary output. The final implementation derives the unique disease target from each semantic mask and supports both train-mode and eval-mode output tuples. The completed smoke-test verifies both fixes.

## Unified evaluator acceptance check

The new evaluator was run on runs4 `best.pt`, val only, using 200 bootstrap draws as a code acceptance check. It reproduced the audited reference values:

- 8-class mIoU: 0.591254
- foreground seven-class mean IoU: 0.538748
- pixel accuracy: 0.925056
- image Accuracy: 0.809176
- image Macro-F1: 0.704120
- `mel -> nv`: 18/167
- `nv -> mel`: 84

It additionally produced lesion-level Macro-F1 0.724283 over 1118 independent lesions, stratified lesion-bootstrap intervals, parameters, GFLOPs and timing. Formal evaluations should use the script default of 2000 bootstrap draws.

Acceptance artifacts are under `verification/runs4_best_val_unified`. Smoke artifacts are under `smoke/loader_and_loss_smoke`.

