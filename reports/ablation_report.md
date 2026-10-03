# Controlled Ablations

All single-factor runs use the same seed, fixed stratified split, 80/20 partition, architecture capacity, optimizer, learning rate, and **200-epoch budget**. Only the named factor changes. Augmentation is applied only through the training transform; validation always uses deterministic resize, tensor conversion, and normalization.

| Controlled change | Macro F1 | Δ vs baseline | Interpretation |
|---|---|---|---|
| CNN baseline | 0.8796 | 0 | reference |
| Augmentation | 0.8265 | -0.0532 | One-factor change from baseline: deterministic resize only |
| Dropout p=0.3 | 0.8733 | -0.0063 | One-factor change from baseline: dropout=0.3 |
| Dropout p=0.5 | 0.8909 | +0.0113 | One-factor change from baseline: dropout=0.5 |
| Average pooling | 0.8516 | -0.0280 | One-factor change from baseline: average pooling in all four pooling blocks |
| Weight decay 1e-4 | 0.8856 | +0.0060 | One-factor change from baseline: AdamW weight_decay=1e-4 |
| StepLR | 0.8810 | +0.0013 | One-factor change from baseline: StepLR; validation is not used to tune test results |
| Validation-selected combination | 0.9015 | +0.0219 | Validation-selected explicit combination: {"augmentation": true, "dropout": 0.5, "scheduler": "step", "weight_decay": 0.0001} |

The best controlled one-factor/individual run is **Dropout p=0.5** at validation macro-F1 **0.8909**. The explicit combined regularized run reaches **0.9015**. Because the validation set has only 80 images (10 per class), small differences should not be over-interpreted; epoch-level curves and per-class support are in `artifacts/results/experiment_results.json`.

Full training/validation loss, LR, train–validation gap, and validation precision/recall/F1 are recorded in `artifacts/results/experiment_results.json`, summarized in `artifacts/results/experiment_summary.csv`, and visualized in `figures/training_curves.png`.
