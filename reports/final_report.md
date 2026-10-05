# Traffic Vehicle Classification — Final Report

## Executive summary

The project implements a leakage-aware data audit, a four-block CNN, a custom balanced batch sampler, CE/BCE comparison, controlled regularization/scheduler ablations, ResNet18 feature extraction and fine-tuning, uncertainty calibration, frozen-test evaluation, unseen-class analysis, and JSON inference.

The validation-selected production model is **`resnet18_fine_tuning`**. On the once-only frozen test evaluation it achieves **85.94% accuracy**, **85.29% macro-F1**, and **85.94% macro recall**. The deployment API uses temperature-scaled Cross-Entropy probabilities and sends confidence below **0.831** to review.

## Required main validation comparison

| Experiment | Macro Precision | Macro Recall | Macro F1 | Lowest-Recall Class | Lowest-Precision Class |
|---|---|---|---|---|---|
| CNN baseline | 0.8894 | 0.8832 | 0.8855 | kamyunet (0.793) | kamyunet (0.719) |
| Imbalanced + balanced batches | 0.7871 | 0.7776 | 0.7749 | kamyun (0.556) | kamyunet (0.579) |
| BCEWithLogitsLoss | 0.9036 | 0.9015 | 0.9016 | kamyun (0.741) | kamyunet (0.781) |
| Best regularized + scheduled | 0.9095 | 0.8992 | 0.9012 | kamyun (0.704) | kamyunet (0.722) |
| ResNet18 feature extraction | 0.9100 | 0.9030 | 0.9026 | kamyun (0.704) | kamyunet (0.781) |
| ResNet18 fine-tuning | 0.9231 | 0.9242 | 0.9231 | kamyun (0.815) | kamyunet (0.844) |

Lowest-class analysis for each meaningful case:

| Experiment | Lowest precision / recall analysis |
|---|---|
| CNN baseline | The lowest recall is **kamyunet (0.793)** and the lowest precision is **kamyunet (0.719)**. These are validation diagnostics; test results were not consulted to choose this run. |
| Imbalanced + balanced batches | The lowest recall is **kamyun (0.556)** and the lowest precision is **kamyunet (0.579)**. These are validation diagnostics; test results were not consulted to choose this run. |
| BCEWithLogitsLoss | The lowest recall is **kamyun (0.741)** and the lowest precision is **kamyunet (0.781)**. These are validation diagnostics; test results were not consulted to choose this run. |
| Best regularized + scheduled | The lowest recall is **kamyun (0.704)** and the lowest precision is **kamyunet (0.722)**. These are validation diagnostics; test results were not consulted to choose this run. |
| ResNet18 feature extraction | The lowest recall is **kamyun (0.704)** and the lowest precision is **kamyunet (0.781)**. These are validation diagnostics; test results were not consulted to choose this run. |
| ResNet18 fine-tuning | The lowest recall is **kamyun (0.815)** and the lowest precision is **kamyunet (0.844)**. These are validation diagnostics; test results were not consulted to choose this run. |

## Complete validation experiment table

| Experiment | Category | Accuracy | Macro P | Macro R | Macro F1 | Best epoch | Seconds |
|---|---|---|---|---|---|---|---|
| CNN baseline | ablation | 0.8836 | 0.8894 | 0.8832 | 0.8855 | 259 | 1165.1 |
| No augmentation | ablation | 0.8233 | 0.8268 | 0.8237 | 0.8227 | 263 | 871.3 |
| Dropout p=0.3 | ablation | 0.8966 | 0.9029 | 0.8918 | 0.8949 | 298 | 1237.3 |
| Dropout p=0.5 | ablation | 0.8966 | 0.9010 | 0.8953 | 0.8975 | 190 | 1205.8 |
| Average pooling | ablation | 0.8879 | 0.8948 | 0.8874 | 0.8893 | 251 | 1187.3 |
| Weight decay 1e-4 | ablation | 0.8966 | 0.9025 | 0.8942 | 0.8969 | 245 | 1316.9 |
| StepLR | ablation | 0.8922 | 0.8938 | 0.8892 | 0.8899 | 118 | 1294.1 |
| BCEWithLogitsLoss | ablation | 0.9009 | 0.9036 | 0.9015 | 0.9016 | 266 | 1280.0 |
| Best regularized + scheduled | main | 0.9009 | 0.9095 | 0.8992 | 0.9012 | 284 | 1261.1 |
| Imbalanced + standard batches | imbalance | 0.8017 | 0.8160 | 0.8064 | 0.7993 | 270 | 1442.2 |
| Imbalanced + balanced batches | imbalance | 0.7759 | 0.7871 | 0.7776 | 0.7749 | 64 | 1293.8 |
| ResNet18 feature extraction | transfer | 0.9052 | 0.9100 | 0.9030 | 0.9026 | 168 | 928.9 |
| ResNet18 fine-tuning | transfer | 0.9224 | 0.9231 | 0.9242 | 0.9231 | 297 | 1003.9 |

## Frozen test result

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 1.000 | 0.875 | 0.933 | 8 |
| autobus | 1.000 | 1.000 | 1.000 | 8 |
| kamyun | 0.700 | 0.875 | 0.778 | 8 |
| kamyunet | 1.000 | 0.500 | 0.667 | 8 |
| minibus | 0.727 | 1.000 | 0.842 | 8 |
| savari | 0.800 | 1.000 | 0.889 | 8 |
| taxi | 1.000 | 1.000 | 1.000 | 8 |
| vanet | 0.833 | 0.625 | 0.714 | 8 |

- Selected model: `resnet18_fine_tuning`
- Validation macro-F1: **0.9231**
- Test accuracy / macro-F1: **0.8594 / 0.8529**
- Test macro precision / recall: **0.8826 / 0.8594**
- Temperature: **2.670** (selected on validation)
- Review threshold: **0.831** (selected on validation)
- Test review rate at frozen threshold: **25.00%**
- Top confusion pair: `kamyun` ↔ `kamyunet` (0.375)

## Key findings

1. **Leakage:** both encoded-file SHA-256 and canonical decoded-pixel hashing identify 0 cross-split duplicate groups in this copy. The canonical check is encoding-independent by design. Frozen test stays intact; 8 pixel-identical train copies quarantined to `dataset/quarantine/` because the same image is in frozen test; 0 duplicate `unclean` copies excluded.
2. **Sampling:** exact balanced batches change the recall/accuracy trade-off under simulated imbalance. Macro-F1 and minority recall should be prioritized over raw accuracy for an equitable classifier.
3. **Loss:** CE models the mutually exclusive task directly and supplies normalized production probabilities. BCE is retained as the required one-vs-all educational comparison; sigmoid scores are not a probability distribution.
4. **Transfer learning:** ImageNet features are valuable with only 320 clean training images. All parameter groups and per-epoch LRs are recorded; test did not decide the strategy.
5. **Reliability:** confidence is imperfect. Validation temperature scaling and a bounded review workload improve the system's operational safety, while `neysan` demonstrates why confidence alone is not OOD detection.

## Detailed reports and figures

- `data_audit.md` — quality checks, dimensions, duplicates, exclusions
- `ablation_report.md` — one-factor and explicit-combination experiments
- `balanced_sampler_report.md` — imbalance indices and per-class comparison
- `loss_comparison.md` — CE/BCE assumptions, curves, metrics
- `transfer_learning_report.md` — ResNet parameters, stages, learning rates
- `error_analysis.md` — test metrics, confusion pairs, threshold, merge decision
- `unclean_analysis.md` — cleaned duplicates and unseen `neysan`
- `figures/representative_images.png` — 16 labeled examples
- `figures/training_curves.png` — per-epoch train/validation behavior and macro-F1
- `figures/final_confusion_matrices.png` — count and row-normalized matrices
- `figures/misclassified_test_images.png` — up to 12 lowest-confidence labeled errors
- `figures/unclean_low_confidence_examples.png` — unseen-class examples

## Final checklist

| Completed item | Evidence |
|---|---|
| ☑ Class mapping is asserted identical for train and test. | Recorded artifact/report |
| ☑ Audit precedes splitting; frozen test retains all 64 images. | Recorded artifact/report |
| ☑ Test is evaluated only after validation selection and is never used for tuning; evidence archives are hash-bound. | Recorded artifact/report |
| ☑ A per-artifacts-directory writer lock prevents concurrent pipeline runs. | Recorded artifact/report |
| ☑ Canonical pixel hashes find 0 cross-split groups. | Recorded artifact/report |
| ☑ Augmentation is confined to training transforms. | Recorded artifact/report |
| ☑ Balanced batches use the recorded reproducible imbalanced subset. | Recorded artifact/report |
| ☑ BCE receives one-hot float targets and predictions use argmax(logits). | Recorded artifact/report |
| ☑ Scheduler receives validation loss/metric only; LR is logged each epoch. | Recorded artifact/report |
| ☑ Per-epoch atomic checkpoints support exact resume; compatible completed checkpoints are reused. | Recorded artifact/report |
| ☑ Accuracy, macro/per-class metrics, and both confusion-matrix forms are saved. | Recorded artifact/report |
| ☑ Per-class metrics are compared across all main experiments. | Recorded artifact/report |
| ☑ Lowest-precision and lowest-recall classes are reported for every result. | Recorded artifact/report |
| ☑ Up to 9 frozen-test errors are plotted with labels and confidence (12 required for the full run). | Recorded artifact/report |
| ☑ unclean is analyzed separately after duplicate removal. | Recorded artifact/report |
| ☑ Final checkpoint stores mapping, transform, threshold, temperature, seed, and config. | Recorded artifact/report |
| ☑ README provides a one-command reproduction path. | Recorded artifact/report |
