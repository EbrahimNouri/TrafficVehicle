# Controlled Ablations

All single-factor runs use the same seed, fixed stratified split, 80/20 partition, architecture capacity, optimizer, learning rate, and **80-epoch budget**. Only the named factor changes. Augmentation is applied only through the training transform; validation always uses deterministic resize, tensor conversion, and normalization.

| Controlled change | Macro F1 | Δ vs baseline | Interpretation |
|---|---|---|---|
| CNN baseline | 0.8458 | 0 | reference |
| Augmentation | 0.8214 | -0.0244 | One-factor change from baseline: deterministic resize only |
| Dropout p=0.3 | 0.8529 | +0.0071 | One-factor change from baseline: dropout=0.3 |
| Dropout p=0.5 | 0.8490 | +0.0032 | One-factor change from baseline: dropout=0.5 |
| Average pooling | 0.8165 | -0.0294 | One-factor change from baseline: average pooling in all four pooling blocks |
| Weight decay 1e-4 | 0.8561 | +0.0103 | One-factor change from baseline: AdamW weight_decay=1e-4 |
| StepLR | 0.8525 | +0.0067 | One-factor change from baseline: StepLR; validation is not used to tune test results |
| Validation-selected combination | 0.8364 | -0.0094 | Validation-selected explicit combination: {"augmentation": true, "dropout": 0.3, "scheduler": "step", "weight_decay": 0.0001} |

The best controlled one-factor/individual run is **Weight decay 1e-4** at validation macro-F1 **0.8561**. The explicit combined regularized run reaches **0.8364**. Because the validation set has only 80 images (10 per class), small differences should not be over-interpreted; epoch-level curves and per-class support are in `artifacts/results/experiment_results.json`.

Full training/validation loss, LR, train–validation gap, and validation precision/recall/F1 are recorded in `artifacts/results/experiment_results.json`, summarized in `artifacts/results/experiment_summary.csv`, and visualized in `figures/training_curves.png`.
