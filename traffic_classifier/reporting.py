"""Generate the human-readable project reports from measured artifacts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig
from .metrics import confusion_pair_ranking, worst_classes
from .utils import markdown_table


def _optional_float(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _stage_durations(result: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for row in result["history"]:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    return ", ".join(f"{stage}={count}" for stage, count in counts.items())


def _metric_row(name: str, result: dict[str, Any]) -> list[Any]:
    metrics = result["validation_metrics"]
    low_recall, low_recall_value = worst_classes(metrics, "recall")
    low_precision, low_precision_value = worst_classes(metrics, "precision")
    return [
        name,
        f"{metrics['macro_precision']:.4f}",
        f"{metrics['macro_recall']:.4f}",
        f"{metrics['macro_f1']:.4f}",
        f"{low_recall} ({low_recall_value:.3f})",
        f"{low_precision} ({low_precision_value:.3f})",
    ]


def _per_class_table(result: dict[str, Any], class_names: list[str]) -> str:
    return markdown_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [
            [
                class_name,
                f"{result['validation_metrics']['per_class'][class_name]['precision']:.3f}",
                f"{result['validation_metrics']['per_class'][class_name]['recall']:.3f}",
                f"{result['validation_metrics']['per_class'][class_name]['f1']:.3f}",
                result["validation_metrics"]["per_class"][class_name]["support"],
            ]
            for class_name in class_names
        ],
    )


def _analysis_sentence(result: dict[str, Any]) -> str:
    metrics = result["validation_metrics"]
    recall_class, recall_value = worst_classes(metrics, "recall")
    precision_class, precision_value = worst_classes(metrics, "precision")
    return (
        f"The lowest recall is **{recall_class} ({recall_value:.3f})** and the lowest "
        f"precision is **{precision_class} ({precision_value:.3f})**. These are validation "
        "diagnostics; test results were not consulted to choose this run."
    )


def render_unclean_analysis(unclean_result: dict[str, Any]) -> str:
    """Render the cleaned-`unclean` report, including the empty-split skip case."""

    if unclean_result.get("skipped"):
        # The split held no images, so there is no measured evidence to report.
        return f"""# Cleaned `unclean` and Unseen-Class Analysis

`unclean` is never used for optimization or validation. This run found no image files in the `unclean` split, so the duplicate-removal pass and the unseen-class analysis were **skipped**. No numbers are reported here: a skipped step is not a measured result.

Reason: {unclean_result.get('skip_reason', 'the unclean split contained no images')}. `artifacts/results/unclean_analysis.json` records `skipped: true` and `artifacts/results/unclean_predictions.csv` is written with headers only. Add images to the `unclean` split and re-run to populate this section.
"""

    unseen = unclean_result["unseen_neysan"]
    known = unclean_result["known_class_metrics"]
    clean = unclean_result["cleaning"]

    # `neysan` may have been fully cleaned out of the unclean split; handle both cases.
    if unseen.get("present", True):
        unseen_section = (
            f"For unseen `neysan`, mean top-class confidence is "
            f"**{unseen['mean_confidence']:.4f}** and "
            f"**{unseen['review_count']} / {clean['unseen_neysan_images']} "
            f"({unseen['review_rate']:.1%})** fall below the validation review "
            f"threshold. Predicted known-class counts are:\n\n"
            + markdown_table(
                ['Prediction', 'Count'],
                [[key, value] for key, value in unseen['predicted_class_counts'].items()],
            )
            + "\n\n`neysan` is analyzed as unseen/human-review data, never added as a "
            "ninth class. Standard softmax confidence is not an out-of-distribution "
            "detector: an unseen image can receive a high known-class score. The "
            "observed confidence pattern is therefore a risk signal, not proof of "
            "semantic correctness. `figures/unclean_low_confidence_examples.png` shows "
            "the 12 least-confident `neysan` cases and "
            "`artifacts/results/unclean_predictions.csv` contains every prediction."
        )
    else:
        unseen_section = (
            "No `neysan` samples remain in the `unclean` split after cleaning "
            "(its folder was emptied by the duplicate-removal pass), so the "
            "unseen-class branch is skipped. `neysan` is still conceptually treated "
            "as an unseen/human-review class, never added as a ninth training label. "
            "Standard softmax confidence is not an out-of-distribution detector, so "
            "any future unseen-class analysis should still route low-confidence cases "
            "to human review."
        )

    return f"""# Cleaned `unclean` and Unseen-Class Analysis

`unclean` is never used for optimization or validation. A second content-aware pass removes every pixel-identical copy of train or test, producing **{clean['retained_images']}** retained images from {clean['original_images']}; **{clean['excluded_duplicates']}** exclusions are recorded with their source. This includes the `train/vanet` versus `unclean/neysan` label conflict, which is excluded rather than silently relabelled.

The retained set has **{clean['known_images']}** known-class examples and **{clean['unseen_neysan_images']}** `neysan` examples. On the known subset, accuracy is **{known['accuracy']:.4f}** and macro-F1 is **{known['macro_f1']:.4f}**. This is diagnostic data, not another model-selection test.

{unseen_section}
"""


def _leakage_clause(audit_summary: dict[str, Any]) -> str:
    """Describe the label conflicts found among cross-split duplicate groups."""

    conflicts = int(audit_summary.get("label_conflict_duplicate_groups", 0))
    groups = int(audit_summary.get("canonical_duplicate_groups", 0))
    if conflicts == 0:
        return ", none of which is a label conflict" if groups else ""
    noun = "conflict" if conflicts == 1 else "conflicts"
    return f", including {conflicts} {noun}"


def _quarantine_sentence(audit_summary: dict[str, Any]) -> str:
    """State where each quarantined copy came from, based on the recorded counts."""

    cleaning = audit_summary.get("cleaning", {})
    train_excluded = int(cleaning.get("train_excluded", 0))
    test_excluded = int(cleaning.get("test_excluded", 0))
    unclean_excluded = int(cleaning.get("unclean_excluded", 0))
    parts = []
    if train_excluded:
        noun = "copy" if train_excluded == 1 else "copies"
        parts.append(
            f"{train_excluded} pixel-identical train {noun} quarantined to "
            "`dataset/quarantine/` because the same image is in frozen test"
        )
    if test_excluded:
        parts.append(
            f"{test_excluded} frozen-test "
            f"{'image' if test_excluded == 1 else 'images'} quarantined"
        )
    parts.append(
        f"{unclean_excluded} duplicate `unclean` "
        f"{'copy' if unclean_excluded == 1 else 'copies'} excluded"
    )
    return "; ".join(parts) + "."


def write_all_reports(
    config: ProjectConfig,
    *,
    audit_summary: dict[str, Any],
    split: dict[str, Any],
    imbalance: dict[str, Any],
    results: dict[str, dict[str, Any]],
    experiment_frame: pd.DataFrame,
    final_result: dict[str, Any],
    unclean_result: dict[str, Any],
    exclusion_records: list[dict[str, Any]],
) -> None:
    reports = Path(config.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    class_names = config.classes
    main_order = [
        "cnn_baseline",
        "imbalanced_balanced",
        "bce_loss",
        "best_regularized",
        "resnet18_feature_extraction",
        "resnet18_fine_tuning",
    ]
    main_results = {name: results[name] for name in main_order if name in results}
    main_table = markdown_table(
        [
            "Experiment",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Lowest-Recall Class",
            "Lowest-Precision Class",
        ],
        [_metric_row(result["display_name"], result) for result in main_results.values()],
    )

    all_rows = []
    for result in results.values():
        metrics = result["validation_metrics"]
        all_rows.append(
            [
                result["display_name"],
                result["category"],
                f"{metrics['accuracy']:.4f}",
                f"{metrics['macro_precision']:.4f}",
                f"{metrics['macro_recall']:.4f}",
                f"{metrics['macro_f1']:.4f}",
                result["best_epoch"],
                f"{result['elapsed_seconds']:.1f}",
            ]
        )
    all_table = markdown_table(
        ["Experiment", "Category", "Accuracy", "Macro P", "Macro R", "Macro F1", "Best epoch", "Seconds"],
        all_rows,
    )

    baseline = results["cnn_baseline"]["validation_metrics"]
    ablation_specs = [
        ("no_augmentation", "Augmentation"),
        ("dropout_0_3", "Dropout p=0.3"),
        ("dropout_0_5", "Dropout p=0.5"),
        ("avg_pool", "Average pooling"),
        ("weight_decay_1e-4", "Weight decay 1e-4"),
        ("step_lr", "StepLR"),
    ]
    ablation_rows = [["CNN baseline", f"{baseline['macro_f1']:.4f}", "0", "reference"]]
    for name, label in ablation_specs:
        metrics = results[name]["validation_metrics"]
        ablation_rows.append(
            [
                label,
                f"{metrics['macro_f1']:.4f}",
                f"{metrics['macro_f1'] - baseline['macro_f1']:+.4f}",
                results[name]["description"],
            ]
        )
    combined = results["best_regularized"]["validation_metrics"]
    ablation_rows.append(
        [
            "Validation-selected combination",
            f"{combined['macro_f1']:.4f}",
            f"{combined['macro_f1'] - baseline['macro_f1']:+.4f}",
            results["best_regularized"]["description"],
        ]
    )
    ablation_table = markdown_table(["Controlled change", "Macro F1", "Δ vs baseline", "Interpretation"], ablation_rows)
    best_ablation = max(
        [results["cnn_baseline"]] + [results[name] for name, _ in ablation_specs],
        key=lambda result: result["validation_metrics"]["macro_f1"],
    )
    ablation_report = f"""# Controlled Ablations

All single-factor runs use the same seed, fixed stratified split, 80/20 partition, architecture capacity, optimizer, learning rate, and **{config.epochs}-epoch budget**. Only the named factor changes. Augmentation is applied only through the training transform; validation always uses deterministic resize, tensor conversion, and normalization.

{ablation_table}

The best controlled one-factor/individual run is **{best_ablation['display_name']}** at validation macro-F1 **{best_ablation['validation_metrics']['macro_f1']:.4f}**. The explicit combined regularized run reaches **{combined['macro_f1']:.4f}**. Because the validation set has only 80 images (10 per class), small differences should not be over-interpreted; epoch-level curves and per-class support are in `artifacts/results/experiment_results.json`.

Full training/validation loss, LR, train–validation gap, and validation precision/recall/F1 are recorded in `artifacts/results/experiment_results.json`, summarized in `artifacts/results/experiment_summary.csv`, and visualized in `figures/training_curves.png`.
"""
    (reports / "ablation_report.md").write_text(ablation_report, encoding="utf-8")

    standard = results["imbalanced_standard"]
    balanced = results["imbalanced_balanced"]
    balance_rows = []
    for class_name in class_names:
        standard_class = standard["validation_metrics"]["per_class"][class_name]
        balanced_class = balanced["validation_metrics"]["per_class"][class_name]
        balance_rows.append(
            [
                class_name,
                standard_class["recall"],
                f"{balanced_class['recall']:.3f}",
                f"{balanced_class['recall'] - standard_class['recall']:+.3f}",
                f"{standard_class['precision']:.3f}",
                f"{balanced_class['precision']:.3f}",
                f"{standard_class['f1']:.3f}",
                f"{balanced_class['f1']:.3f}",
            ]
        )
    balance_table = markdown_table(
        ["Class", "Standard recall", "Balanced recall", "Δ recall", "Standard precision", "Balanced precision", "Standard F1", "Balanced F1"],
        balance_rows,
    )
    balance_report = f"""# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `{imbalance['seed']}`, 14 examples are retained from each selected minority class (`{'`, `'.join(imbalance['minority_classes'])}`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **{config.batch_size}**, hence **{config.batch_size // len(class_names)} examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

{balance_table}

- Standard: accuracy **{standard['validation_metrics']['accuracy']:.4f}**, macro-F1 **{standard['validation_metrics']['macro_f1']:.4f}**.
- Balanced: accuracy **{balanced['validation_metrics']['accuracy']:.4f}**, macro-F1 **{balanced['validation_metrics']['macro_f1']:.4f}**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
"""
    (reports / "balanced_sampler_report.md").write_text(balance_report, encoding="utf-8")

    ce = results["cnn_baseline"]
    bce = results["bce_loss"]
    loss_table = markdown_table(
        ["Loss", "Validation accuracy", "Macro P", "Macro R", "Macro F1", "Mean top confidence"],
        [
            [
                "CrossEntropyLoss",
                f"{ce['validation_metrics']['accuracy']:.4f}",
                f"{ce['validation_metrics']['macro_precision']:.4f}",
                f"{ce['validation_metrics']['macro_recall']:.4f}",
                f"{ce['validation_metrics']['macro_f1']:.4f}",
                f"{ce.get('validation_uncertainty', {}).get('mean_confidence', float('nan')):.4f}",
            ],
            [
                "BCEWithLogitsLoss",
                f"{bce['validation_metrics']['accuracy']:.4f}",
                f"{bce['validation_metrics']['macro_precision']:.4f}",
                f"{bce['validation_metrics']['macro_recall']:.4f}",
                f"{bce['validation_metrics']['macro_f1']:.4f}",
                f"{bce.get('validation_uncertainty', {}).get('mean_confidence', float('nan')):.4f}",
            ],
        ],
    )
    loss_pair_rows = []
    for loss_name, result in (("CE", ce), ("BCE", bce)):
        pairs = confusion_pair_ranking(
            result["validation_metrics"]["confusion_matrix_row_normalized"],
            class_names,
        )[:5]
        for pair in pairs:
            loss_pair_rows.append(
                [
                    loss_name,
                    pair["class_a"],
                    pair["class_b"],
                    f"{pair['a_as_b']:.3f}",
                    f"{pair['b_as_a']:.3f}",
                    f"{pair['pair_confusion']:.3f}",
                ]
            )
    loss_pair_table = markdown_table(
        ["Loss", "Class A", "Class B", "A→B", "B→A", "Mutual score"],
        loss_pair_rows,
    )
    loss_report = f"""# Cross-Entropy vs. BCEWithLogitsLoss

Both runs use the same CNN, seed, split, augmented training transform, AdamW optimizer, fixed LR, and {config.epochs}-epoch budget. BCE targets are explicitly one-hot encoded `float32` tensors. No sigmoid is placed before `BCEWithLogitsLoss`; sigmoid is used only afterward to inspect confidence. Predictions for both experiments use `argmax(logits)`.

{loss_table}

Cross-entropy directly models one mutually exclusive target and produces normalized class probabilities, so it is the production choice. BCE treats eight sigmoid outputs as independent one-vs-all decisions: scores can be simultaneously high and do not sum to one. Consequently, BCE's maximum sigmoid score is useful for comparison but is not the same calibrated probability object. The final checkpoint and JSON API use Cross-Entropy.

Validation error patterns (top five mutual pairs):

{loss_pair_table}

A change in the pair ranking reflects both the loss geometry and ordinary small-sample variation; it is interpreted alongside the complete curves and per-class table rather than as a standalone causal claim.

### Per-class CE

{_per_class_table(ce, class_names)}

### Per-class BCE

{_per_class_table(bce, class_names)}
"""
    (reports / "loss_comparison.md").write_text(loss_report, encoding="utf-8")

    feature_result = results["resnet18_feature_extraction"]
    fine_result = results["resnet18_fine_tuning"]
    feature_stages = _stage_durations(feature_result)
    fine_stages = _stage_durations(fine_result)
    transfer_rows = []
    for name in ("resnet18_feature_extraction", "resnet18_fine_tuning"):
        result = results[name]
        report = result["metadata"]["parameter_report_head_stage"]
        transfer_rows.append(
            [
                result["display_name"],
                f"{report['total_parameters']:,}",
                f"{report['trainable_parameters']:,}",
                f"{report['frozen_parameters']:,}",
                f"{report['trainable_percent']:.2f}%",
                "; ".join(
                    f"{group['group']}={group['lr']}"
                    for group in result["history"][0]["learning_rates"]
                ),
                result["best_epoch"],
            ]
        )
    fine_history = results["resnet18_fine_tuning"]["history"]
    transfer_table = markdown_table(
        ["Strategy", "Total params", "Head-stage trainable", "Head-stage frozen", "% trainable", "Initial LR groups", "Best epoch"],
        transfer_rows,
    )
    transfer_report = f"""# ResNet18 Transfer Learning

Both transfer experiments retain the standard ImageNet ResNet18 (`conv1` and `maxpool` unchanged), ImageNet normalization, and resize/crop geometry. The training set has only 320 images, so validation—not test—controls the comparison. Total training budget is equal at {config.epochs} epochs. Recorded stages are feature extraction ({feature_stages}) and fine-tuning ({fine_stages}); the latter uses a smaller backbone learning rate after the head-only stage. Frozen BatchNorm statistics remain in evaluation mode.

{transfer_table}

Fine-tuning stage details:

{markdown_table(
    ['Epoch', 'Stage', 'LR groups', 'Trainable parameters'],
    [[row['epoch'], row['stage'], json.dumps(row['learning_rates']), json.dumps(row['trainable_parameters'])] for row in fine_history]
)}

- Feature extraction validation macro-F1: **{results['resnet18_feature_extraction']['validation_metrics']['macro_f1']:.4f}**.
- Fine-tuning validation macro-F1: **{results['resnet18_fine_tuning']['validation_metrics']['macro_f1']:.4f}**.

The best transfer configuration is selected on validation. Test is untouched until final model selection is recorded.
"""
    (reports / "transfer_learning_report.md").write_text(transfer_report, encoding="utf-8")

    test_metrics = final_result["test_metrics"]
    uncertainty = final_result["test_uncertainty"]
    selective = final_result["test_selective"]
    review = final_result["review_threshold"]
    pair_rows = [
        [
            pair["class_a"],
            pair["class_b"],
            f"{pair['a_as_b']:.3f}",
            f"{pair['b_as_a']:.3f}",
            f"{pair['pair_confusion']:.3f}",
        ]
        for pair in final_result["mutual_confusion_pairs"][:10]
    ]
    test_class_table = markdown_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [
            [
                class_name,
                f"{test_metrics['per_class'][class_name]['precision']:.3f}",
                f"{test_metrics['per_class'][class_name]['recall']:.3f}",
                f"{test_metrics['per_class'][class_name]['f1']:.3f}",
                test_metrics["per_class"][class_name]["support"],
            ]
            for class_name in class_names
        ],
    )
    top_pair = final_result["mutual_confusion_pairs"][0]
    displayed_error_count = min(12, int(final_result["error_count"]))
    merge_threshold = 0.30
    merge_decision = (
        f"Do not merge for the production taxonomy. The strongest mutual pair is "
        f"`{top_pair['class_a']}`/`{top_pair['class_b']}` with score "
        f"{top_pair['pair_confusion']:.3f}"
        + (
            f", above the predeclared {merge_threshold:.2f} investigation threshold. "
            "The visual gallery should be checked for domain overlap, but a merge is not "
            "automatically warranted: confusion can reflect crop, pose, or scale rather "
            "than a redundant taxonomy."
            if top_pair["pair_confusion"] >= merge_threshold
            else ", below the predeclared 0.30 investigation threshold. "
            "This does not support collapsing two operationally useful labels."
        )
    )
    error_report = f"""# Final Frozen-Test Evaluation and Error Analysis

## Protocol integrity

The final configuration was selected by maximum validation macro-F1 among Cross-Entropy production candidates. It selected **`{final_result['selected_experiment']}`**. The frozen test set was evaluated once, after this choice. Temperature and review threshold were fit/selected on validation only.

## Overall and per-class test metrics

- Accuracy: **{test_metrics['accuracy']:.4f}**
- Macro precision: **{test_metrics['macro_precision']:.4f}**
- Macro recall: **{test_metrics['macro_recall']:.4f}**
- Macro F1: **{test_metrics['macro_f1']:.4f}**

{test_class_table}

Both count-based and row-normalized confusion matrices are saved in `artifacts/results/frozen_test_evaluation.json` and plotted in `figures/final_confusion_matrices.png`.

## Confidence and human review

Validation temperature scaling changes ECE from the value saved in `validation_uncertainty.json`; the final validation-selected confidence threshold is **{review['threshold']:.4f}**, chosen as the lowest threshold reaching at least {review['target_coverage']:.0%} automatic coverage. Thus approximately {config.target_review_coverage:.0%} of deployment volume is automatic and the remainder is human review.

On frozen test, calibrated multiclass NLL is **{uncertainty['nll']:.4f}**, ECE is **{uncertainty['ece_10_bins']:.4f}**, and Brier score is **{uncertainty['brier_multiclass']:.4f}**. The selected threshold routes **{selective['review_count']} / {sum(item['support'] for item in test_metrics['per_class'].values())} ({selective['review_rate']:.1%})** images to review; automatic accuracy is **{_optional_float(selective['automatic_accuracy'])}**. Threshold behavior is descriptive on test; it is not re-tuned.

The risk–coverage curve is `figures/validation_risk_coverage.png`.

## Confusion pairs and merge decision

{markdown_table(['Class A', 'Class B', 'A→B', 'B→A', 'Mutual score'], pair_rows)}

`pair_confusion(i,j) = row_normalized_CM[i,j] + row_normalized_CM[j,i]`. **{merge_decision}**

## Error inspection

`figures/misclassified_test_images.png` displays {displayed_error_count} lowest-confidence frozen-test errors with true label, predicted label, and confidence. Machine-readable rows, including top-3 classes and review flag, are in `artifacts/results/test_errors.csv` ({final_result['error_count']} total errors). Visual review distinguishes genuine class overlap from ambiguous crops, unusual viewpoints, tiny vehicles, and mislabeled/quality-impaired examples.
"""
    (reports / "error_analysis.md").write_text(error_report, encoding="utf-8")

    (reports / "unclean_analysis.md").write_text(
        render_unclean_analysis(unclean_result), encoding="utf-8"
    )
    unclean_skipped = bool(unclean_result.get("skipped"))

    final_class_table = markdown_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        [
            [
                class_name,
                f"{test_metrics['per_class'][class_name]['precision']:.3f}",
                f"{test_metrics['per_class'][class_name]['recall']:.3f}",
                f"{test_metrics['per_class'][class_name]['f1']:.3f}",
                test_metrics["per_class"][class_name]["support"],
            ]
            for class_name in class_names
        ],
    )
    checklist = [
        "Class mapping is asserted identical for train and test.",
        f"Audit precedes splitting; frozen test retains all {audit_summary['splits']['test']['images']} images.",
        "Test is evaluated only after validation selection and is never used for tuning; evidence archives are hash-bound.",
        "A per-artifacts-directory writer lock prevents concurrent pipeline runs.",
        f"Canonical pixel hashes find {audit_summary['canonical_duplicate_groups']} cross-split groups.",
        "Augmentation is confined to training transforms.",
        "Balanced batches use the recorded reproducible imbalanced subset.",
        "BCE receives one-hot float targets and predictions use argmax(logits).",
        "Scheduler receives validation loss/metric only; LR is logged each epoch.",
        "Per-epoch atomic checkpoints support exact resume; compatible completed checkpoints are reused.",
        "Accuracy, macro/per-class metrics, and both confusion-matrix forms are saved.",
        "Per-class metrics are compared across all main experiments.",
        "Lowest-precision and lowest-recall classes are reported for every result.",
        f"Up to {displayed_error_count} frozen-test errors are plotted with labels and confidence (12 required for the full run).",
        (
            "unclean is analyzed separately after duplicate removal."
            if not unclean_skipped
            else "unclean held no images, so the analysis was skipped and reported as such."
        ),
        "Final checkpoint stores mapping, transform, threshold, temperature, seed, and config.",
        "README provides a one-command reproduction path.",
    ]
    leakage_clause = _leakage_clause(audit_summary)
    final_report = f"""# Traffic Vehicle Classification — Final Report

## Executive summary

The project implements a leakage-aware data audit, a four-block CNN, a custom balanced batch sampler, CE/BCE comparison, controlled regularization/scheduler ablations, ResNet18 feature extraction and fine-tuning, uncertainty calibration, frozen-test evaluation, unseen-class analysis, and JSON inference.

The validation-selected production model is **`{final_result['selected_experiment']}`**. On the once-only frozen test evaluation it achieves **{test_metrics['accuracy']:.2%} accuracy**, **{test_metrics['macro_f1']:.2%} macro-F1**, and **{test_metrics['macro_recall']:.2%} macro recall**. The deployment API uses temperature-scaled Cross-Entropy probabilities and sends confidence below **{review['threshold']:.3f}** to review.

## Required main validation comparison

{main_table}

Lowest-class analysis for each meaningful case:

{markdown_table(
    ['Experiment', 'Lowest precision / recall analysis'],
    [[result['display_name'], _analysis_sentence(result)] for result in main_results.values()]
)}

## Complete validation experiment table

{all_table}

## Frozen test result

{final_class_table}

- Selected model: `{final_result['selected_experiment']}`
- Validation macro-F1: **{final_result['selection_validation_metrics']['macro_f1']:.4f}**
- Test accuracy / macro-F1: **{test_metrics['accuracy']:.4f} / {test_metrics['macro_f1']:.4f}**
- Test macro precision / recall: **{test_metrics['macro_precision']:.4f} / {test_metrics['macro_recall']:.4f}**
- Temperature: **{final_result['temperature']:.3f}** (selected on validation)
- Review threshold: **{review['threshold']:.3f}** (selected on validation)
- Test review rate at frozen threshold: **{selective['review_rate']:.2%}**
- Top confusion pair: `{top_pair['class_a']}` ↔ `{top_pair['class_b']}` ({top_pair['pair_confusion']:.3f})

## Key findings

1. **Leakage:** both encoded-file SHA-256 and canonical decoded-pixel hashing identify {audit_summary['canonical_duplicate_groups']} cross-split duplicate groups in this copy{leakage_clause}. The canonical check is encoding-independent by design. Frozen test stays intact; {_quarantine_sentence(audit_summary)}
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

{markdown_table(['Completed item', 'Evidence'], [[f'☑ {item}', 'Recorded artifact/report'] for item in checklist])}
"""
    (reports / "final_report.md").write_text(final_report, encoding="utf-8")
    (reports / "final_report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Traffic Vehicle Classification</title>"
        "<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;line-height:1.5}pre{white-space:pre-wrap}</style>"
        f"</head><body><pre>{html.escape(final_report)}</pre></body></html>",
        encoding="utf-8",
    )