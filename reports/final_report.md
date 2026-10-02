# Traffic Vehicle Classification — Final Report

## Executive summary

The project implements a leakage-aware data audit, a four-block CNN, a custom balanced batch sampler, CE/BCE comparison, controlled regularization/scheduler ablations, ResNet18 feature extraction and fine-tuning, uncertainty calibration, frozen-test evaluation, unseen-class analysis, and JSON inference.

The validation-selected production model is **`resnet18_fine_tuning`**. On the once-only frozen test evaluation it achieves **89.75% accuracy**, **89.68% macro-F1**, and **89.75% macro recall**. The deployment API uses temperature-scaled Cross-Entropy probabilities and sends confidence below **0.830** to review.

## Required main validation comparison

| Experiment | Macro Precision | Macro Recall | Macro F1 | Lowest-Recall Class | Lowest-Precision Class |
|---|---|---|---|---|---|
| CNN baseline | 0.7918 | 0.7551 | 0.7477 | kamyunet (0.381) | minibus (0.567) |
| Imbalanced + balanced batches | 0.6277 | 0.6176 | 0.6089 | kamyun (0.421) | autobus (0.448) |
| BCEWithLogitsLoss | 0.7529 | 0.7040 | 0.7036 | ambulance (0.353) | kamyunet (0.469) |
| Best regularized + scheduled | 0.7918 | 0.7551 | 0.7477 | kamyunet (0.381) | minibus (0.567) |
| ResNet18 feature extraction | 0.8625 | 0.8312 | 0.8336 | taxi (0.579) | savari (0.633) |
| ResNet18 fine-tuning | 0.9295 | 0.9279 | 0.9281 | taxi (0.895) | kamyun (0.857) |

Lowest-class analysis for each meaningful case:

| Experiment | Lowest precision / recall analysis |
|---|---|
| CNN baseline | The lowest recall is **kamyunet (0.381)** and the lowest precision is **minibus (0.567)**. These are validation diagnostics; test results were not consulted to choose this run. |
| Imbalanced + balanced batches | The lowest recall is **kamyun (0.421)** and the lowest precision is **autobus (0.448)**. These are validation diagnostics; test results were not consulted to choose this run. |
| BCEWithLogitsLoss | The lowest recall is **ambulance (0.353)** and the lowest precision is **kamyunet (0.469)**. These are validation diagnostics; test results were not consulted to choose this run. |
| Best regularized + scheduled | The lowest recall is **kamyunet (0.381)** and the lowest precision is **minibus (0.567)**. These are validation diagnostics; test results were not consulted to choose this run. |
| ResNet18 feature extraction | The lowest recall is **taxi (0.579)** and the lowest precision is **savari (0.633)**. These are validation diagnostics; test results were not consulted to choose this run. |
| ResNet18 fine-tuning | The lowest recall is **taxi (0.895)** and the lowest precision is **kamyun (0.857)**. These are validation diagnostics; test results were not consulted to choose this run. |

## Complete validation experiment table

| Experiment | Category | Accuracy | Macro P | Macro R | Macro F1 | Best epoch | Seconds |
|---|---|---|---|---|---|---|---|
| CNN baseline | ablation | 0.7590 | 0.7918 | 0.7551 | 0.7477 | 25 | 182.2 |
| No augmentation | ablation | 0.6928 | 0.7189 | 0.6967 | 0.6771 | 19 | 168.4 |
| Dropout p=0.3 | ablation | 0.7048 | 0.7661 | 0.7038 | 0.7043 | 24 | 45.5 |
| Dropout p=0.5 | ablation | 0.6627 | 0.6835 | 0.6607 | 0.6568 | 17 | 49.6 |
| Average pooling | ablation | 0.6265 | 0.6973 | 0.6171 | 0.6191 | 24 | 101.8 |
| Weight decay 1e-4 | ablation | 0.7229 | 0.7667 | 0.7124 | 0.7128 | 24 | 101.4 |
| StepLR | ablation | 0.6446 | 0.6809 | 0.6372 | 0.6362 | 25 | 169.4 |
| BCEWithLogitsLoss | ablation | 0.7108 | 0.7529 | 0.7040 | 0.7036 | 19 | 120.2 |
| Best regularized + scheduled | main | 0.7590 | 0.7918 | 0.7551 | 0.7477 | 25 | 100.3 |
| Imbalanced + standard batches | imbalance | 0.5723 | 0.4403 | 0.5641 | 0.4876 | 24 | 102.7 |
| Imbalanced + balanced batches | imbalance | 0.6084 | 0.6277 | 0.6176 | 0.6089 | 18 | 102.9 |
| ResNet18 feature extraction | transfer | 0.8373 | 0.8625 | 0.8312 | 0.8336 | 25 | 161.1 |
| ResNet18 fine-tuning | transfer | 0.9277 | 0.9295 | 0.9279 | 0.9281 | 24 | 162.6 |

## Frozen test result

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ambulance | 0.979 | 0.920 | 0.948 | 50 |
| autobus | 0.925 | 0.980 | 0.951 | 50 |
| kamyun | 0.714 | 0.800 | 0.755 | 50 |
| kamyunet | 0.829 | 0.680 | 0.747 | 50 |
| minibus | 0.904 | 0.940 | 0.922 | 50 |
| savari | 0.979 | 0.920 | 0.948 | 50 |
| taxi | 0.907 | 0.980 | 0.942 | 50 |
| vanet | 0.960 | 0.960 | 0.960 | 50 |

- Selected model: `resnet18_fine_tuning`
- Validation macro-F1: **0.9281**
- Test accuracy / macro-F1: **0.8975 / 0.8968**
- Test macro precision / recall: **0.8996 / 0.8975**
- Temperature: **1.460** (selected on validation)
- Review threshold: **0.830** (selected on validation)
- Test review rate at frozen threshold: **19.50%**
- Top confusion pair: `kamyun` ↔ `kamyunet` (0.360)

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
| ☑ Audit precedes splitting; frozen test retains all 400 images. | Recorded artifact/report |
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
| ☑ Up to 12 frozen-test errors are plotted with labels and confidence (12 required for the full run). | Recorded artifact/report |
| ☑ unclean held no images, so the analysis was skipped and reported as such. | Recorded artifact/report |
| ☑ Final checkpoint stores mapping, transform, threshold, temperature, seed, and config. | Recorded artifact/report |
| ☑ README provides a one-command reproduction path. | Recorded artifact/report |
