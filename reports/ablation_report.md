# Controlled Ablations

All single-factor runs use the same seed, fixed stratified split, 80/20 partition, architecture capacity, optimizer, learning rate, and **300-epoch budget**. Only the named factor changes. Augmentation is applied only through the training transform; validation always uses deterministic resize, tensor conversion, and normalization.

| Controlled change | Macro F1 | Δ vs baseline | Interpretation |
|---|---|---|---|
| CNN baseline | 0.8855 | 0 | reference |
| Augmentation | 0.8227 | -0.0629 | One-factor change from baseline: deterministic resize only |
| Dropout p=0.3 | 0.8949 | +0.0093 | One-factor change from baseline: dropout=0.3 |
| Dropout p=0.5 | 0.8975 | +0.0120 | One-factor change from baseline: dropout=0.5 |
| Average pooling | 0.8893 | +0.0037 | One-factor change from baseline: average pooling in all four pooling blocks |
| Weight decay 1e-4 | 0.8969 | +0.0114 | One-factor change from baseline: AdamW weight_decay=1e-4 |
| StepLR | 0.8899 | +0.0044 | One-factor change from baseline: StepLR; validation is not used to tune test results |
| Validation-selected combination | 0.9012 | +0.0157 | Validation-selected explicit combination: {"augmentation": true, "dropout": 0.5, "scheduler": "step", "weight_decay": 0.0001} |

The best controlled one-factor/individual run is **Dropout p=0.5** at validation macro-F1 **0.8975**. The explicit combined regularized run reaches **0.9012**. Because the validation set has only 80 images (10 per class), small differences should not be over-interpreted; epoch-level curves and per-class support are in `artifacts/results/experiment_results.json`.

Full training/validation loss, LR, train–validation gap, and validation precision/recall/F1 are recorded in `artifacts/results/experiment_results.json`, summarized in `artifacts/results/experiment_summary.csv`, and visualized in `figures/training_curves.png`.
