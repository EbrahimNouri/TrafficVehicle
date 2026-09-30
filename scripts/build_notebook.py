"""Build the completed project notebook while preserving the original rubric."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "Project-Definition2-traffic-vehicle.ipynb"
original = nbf.read(NOTEBOOK_PATH, as_version=4)
markdown_sources = [
    cell.source for cell in original.cells if cell.cell_type == "markdown"
]
appendix_marker = "# Appendix: Original Project Definition and Grading Rubric"
appendix_index = next(
    (
        index
        for index, source in enumerate(markdown_sources)
        if source.startswith(appendix_marker)
    ),
    None,
)
# Re-running this builder must not recursively append an earlier completed notebook.
original_markdown = (
    markdown_sources[appendix_index + 1 :]
    if appendix_index is not None
    else markdown_sources
)

notebook = nbf.v4.new_notebook()
cells = []
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def add(cell):
    cells.append(cell)


add(md(r"""# Project 2: Traffic Vehicle Classification — Completed

## Reliable traffic-camera classification with CNNs, transfer learning, and human review

This notebook is the executable companion to the original project definition. It audits the data before splitting, prevents cross-split leakage, performs the required controlled experiments, selects a production model on validation data, evaluates the frozen test set once, analyzes unseen/uncertain cases, and provides review-aware JSON inference.

> **Outcome:** a complete reproducible project, not a demonstration with an unevaluated training loop. All measured values below come from saved artifacts in `artifacts/` and `reports/`.

Start with `README.md` for setup and `reports/final_report.md` for the complete written analysis."""))

add(md(r"""## 1. Reproducible setup

The code imports the tested implementation from `traffic_classifier/`. If all result files are absent, the workflow cell trains every experiment. Otherwise it reuses completed checkpoints/results and presents the saved evidence. A per-artifacts-directory OS lock prevents two pipeline writers from publishing conflicting results."""))

add(code(r'''from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import torch
import torchvision
import sklearn
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / "traffic_classifier").exists():
    ROOT = ROOT.parent

required = [
    ROOT / "artifacts/results/experiment_results.json",
    ROOT / "artifacts/results/frozen_test_evaluation.json",
    ROOT / "artifacts/checkpoints/final_model.pt",
]
print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "sklearn": sklearn.__version__,
    "cuda_available": torch.cuda.is_available(),
    "project_root": str(ROOT),
})'''))

add(md(r"""## 2. Audit before validation and before model fitting

The audit verifies counts, formats, readability, dimensions, quality issues, encoded-file hashes, and **decoded RGB pixel hashes**. Both hash methods identify the same duplicate groups in this supplied copy; the canonical pixel hash is encoding-independent by design. No train or test item is silently removed; duplicate copies inside `unclean` are excluded from the separate data-quality analysis."""))

add(code(r'''from traffic_classifier.pipeline import run_project

if not all(path.exists() for path in required):
    run_project(ROOT / "configs/default.json")

audit_summary = json.loads((ROOT / "artifacts/audit/summary.json").read_text(encoding="utf-8"))
audit_frame = pd.DataFrame([
    {
        "split": split,
        "images": values["images"],
        "width_min": values["width_quantiles"]["min"],
        "width_median": values["width_quantiles"]["median"],
        "width_max": values["width_quantiles"]["max"],
        "height_min": values["height_quantiles"]["min"],
        "height_median": values["height_quantiles"]["median"],
        "height_max": values["height_quantiles"]["max"],
        "unique_pixels": values["unique_pixel_sha256"],
    }
    for split, values in audit_summary["splits"].items()
])
display(audit_frame)
print("Class counts:", {k: v["class_counts"] for k, v in audit_summary["splits"].items()})
print("Quality findings:", audit_summary["issue_counts"] or "No unreadable, unsupported, unusually small, or extreme-aspect files")
print("Canonical duplicate groups:", audit_summary["canonical_duplicate_groups"])
print("Cross-split duplicate groups:", audit_summary["cross_split_duplicate_groups"])
print("Label-conflict groups:", audit_summary["label_conflict_duplicate_groups"])
print("Clean counts:", audit_summary["cleaning"]["clean_counts"])'''))

add(code(r'''display(Image(filename=str(ROOT / "reports/figures/representative_images.png"), width=950))'''))
add(md("The gallery contains 16 reproducible examples—two from each known class—and is generated from the exact paths saved in `artifacts/audit/representative_paths.json`."))

add(code(r'''display(Image(filename=str(ROOT / "reports/figures/image_size_distributions.png"), width=950))'''))

add(md(r"""## 3. Leakage-safe split, transforms, and reproducibility

The split is stratified by class with seed 42. Exact sample indices and paths are stored before training. Augmentation is present only in the training transform; validation/test use deterministic resize, `ToTensor`, and ImageNet normalization."""))

add(code(r'''split = json.loads((ROOT / "artifacts/splits/train_validation.json").read_text(encoding="utf-8"))
print("Seed:", split["seed"])
print("Validation fraction:", split["validation_fraction"])
print("Train / validation:", split["train_count"], "/", split["validation_count"])
print("Class mapping:", split["class_to_idx"])
print("Frozen test:", split["frozen_test"])
print("Training transform:", json.dumps(split["transform"], indent=2))
for record in split["by_class"]:
    print(record["class_name"], "train:", len(record["train_indices"]), "validation:", len(record["validation_indices"]))'''))

add(md(r"""## 4. CNN baseline, complete loop, and resumable checkpoints

`TrafficCNN` has four convolution–BatchNorm–ReLU blocks, pooling, adaptive average pooling, optional dropout, and an eight-class linear head. Checkpoints are selected by validation macro-F1. Every epoch records training/validation loss, train–validation gap, validation accuracy/precision/recall/F1, stage, and all learning-rate groups.

Training is restart-safe. After every epoch, `<experiment>.last.pt` is atomically replaced with current model, optimizer, scheduler, best weights, history, elapsed time, and RNG/DataLoader-generator state. Re-running the same pipeline command resumes at the next epoch. A compatible completed best checkpoint is reused without retraining; `--force` explicitly starts fresh."""))

add(code(r'''from traffic_classifier.models import TrafficCNN
from traffic_classifier.data import build_transforms
from traffic_classifier.config import ProjectConfig

config = ProjectConfig.from_json(ROOT / "configs/default.json")
train_transform, train_transform_spec = build_transforms(config, augmentation=True, kind="cnn")
eval_transform, eval_transform_spec = build_transforms(config, augmentation=False, kind="cnn")
print("Train transform:\n", train_transform)
print("Eval transform:\n", eval_transform)

model = TrafficCNN(num_classes=8, dropout=0.3, pooling="max")
print(model)
print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))'''))

add(code(r'''from traffic_classifier.engine import _rolling_checkpoint_path

checkpoint_examples = [
    ROOT / "artifacts/checkpoints/cnn_baseline.pt",
    ROOT / "artifacts/checkpoints/cnn_baseline.last.pt",
]
for path in checkpoint_examples:
    print(path.name, "exists:", path.exists(), "size:", path.stat().st_size if path.exists() else None)
print("Rolling path helper:", _rolling_checkpoint_path(checkpoint_examples[0]).name)'''))

add(md(r"""## 5. Custom `BalancedBatchSampler`

The original training data are balanced, so a reproducible simulated imbalance retains 14 examples from `kamyun`, `minibus`, `savari`, and `vanet`, while retaining 40 from each other class. The sampler oversamples to equal class exposure and places exactly four examples of every class in every size-32 batch. Standard and balanced loaders receive the same number of optimizer steps."""))

add(code(r'''from traffic_classifier.data import (
    load_image_folder, stratified_validation_indices, simulated_imbalance_indices,
    BalancedBatchSampler,
)

base_transform, _ = build_transforms(config, augmentation=True, kind="cnn")
base_train = load_image_folder(ROOT / "dataset", "train", base_transform, config.classes)
train_idx, val_idx, _ = stratified_validation_indices(base_train, 0.20, config.seed)
imbalanced_idx, imbalance_detail = simulated_imbalance_indices(
    train_idx, base_train.targets, base_train.class_to_idx,
    config.imbalanced_classes, config.imbalanced_retained_per_minority_class, config.seed + 100,
)
sampler = BalancedBatchSampler(base_train.targets, imbalanced_idx, 8, 32, seed=config.seed)
sampler.set_epoch(0)
first_batch = next(iter(sampler))
counts = np.bincount(np.asarray(base_train.targets)[first_batch], minlength=8)
print("Retained class counts:", imbalance_detail["class_counts"])
print("Batches / examples per epoch:", len(sampler), sampler.total_size)
print("First batch class counts:", dict(zip(config.classes, counts)))
print("First batch size:", len(first_batch))
print("Exact retained indices are saved in artifacts/splits/simulated_imbalance.json")'''))

add(md(r"""## 6. All controlled validation experiments

Single-factor comparisons use the same seed, split, architecture capacity, optimizer, LR, and epoch budget. The explicit regularized combination is identified as such. The frozen test is not involved in any row below."""))

add(code(r'''experiment_results = json.loads((ROOT / "artifacts/results/experiment_results.json").read_text(encoding="utf-8"))
experiment_summary = pd.read_csv(ROOT / "artifacts/results/experiment_summary.csv")
display(experiment_summary)'''))

add(md(r"""### Required main comparison

The CE/BCE row below uses the BCE result only to identify that controlled loss comparison; the dedicated CE baseline appears in the baseline row. A complete side-by-side CE/BCE table follows."""))

add(code(r'''from traffic_classifier.metrics import worst_classes

main_order = [
    "cnn_baseline", "imbalanced_balanced", "bce_loss", "best_regularized",
    "resnet18_feature_extraction", "resnet18_fine_tuning",
]
main_rows = []
for name in main_order:
    metrics = experiment_results[name]["validation_metrics"]
    low_recall = worst_classes(metrics, "recall")
    low_precision = worst_classes(metrics, "precision")
    main_rows.append({
        "Experiment": experiment_results[name]["display_name"],
        "Macro Precision": metrics["macro_precision"],
        "Macro Recall": metrics["macro_recall"],
        "Macro F1": metrics["macro_f1"],
        "Lowest-Recall Class": f"{low_recall[0]} ({low_recall[1]:.3f})",
        "Lowest-Precision Class": f"{low_precision[0]} ({low_precision[1]:.3f})",
    })
display(pd.DataFrame(main_rows).round(4))'''))

add(code(r'''display(Image(filename=str(ROOT / "reports/figures/training_curves.png"), width=1000))'''))

add(md(r"""### Per-class comparison

This heatmap-style table contains validation precision, recall, and F1 for all main experiments."""))

add(code(r'''display(Image(filename=str(ROOT / "reports/figures/per_class_comparison.png"), width=1000))'''))

add(md(r"""## 7. Balanced-batch effect on every class

Macro-F1 and per-class recall are the primary fairness measures. Overall accuracy can fall because a balanced sampler intentionally sacrifices majority prevalence."""))

add(code(r'''balanced_comparison = json.loads((ROOT / "artifacts/results/balanced_batch_comparison.json").read_text(encoding="utf-8"))
balance_rows = []
for class_name in config.classes:
    standard = balanced_comparison["standard"]["per_class"][class_name]
    balanced = balanced_comparison["balanced"]["per_class"][class_name]
    balance_rows.append({
        "class": class_name,
        "standard_recall": standard["recall"],
        "balanced_recall": balanced["recall"],
        "recall_delta": balanced["recall"] - standard["recall"],
        "standard_precision": standard["precision"],
        "balanced_precision": balanced["precision"],
        "standard_f1": standard["recall"] * 0 + standard["f1"],
        "balanced_f1": balanced["f1"],
    })
display(pd.DataFrame(balance_rows).round(4))
print("Standard macro-F1:", balanced_comparison["standard"]["macro_f1"])
print("Balanced macro-F1:", balanced_comparison["balanced"]["macro_f1"])
print("Equal examples/class/batch:", balanced_comparison["batch_size"] // 8)
print("Same steps/epoch:", balanced_comparison["standard_and_balanced_steps_per_epoch"])'''))

add(md(r"""## 8. Cross-entropy vs. BCEWithLogitsLoss

BCE uses one-hot `float32` targets, no pre-loss sigmoid, and `argmax(logits)` for classification. Cross-entropy models the mutually exclusive task directly and is the only eligible production loss. BCE sigmoid scores are independent and do not form a normalized probability distribution."""))

add(code(r'''loss_rows = []
for name in ("cnn_baseline", "bce_loss"):
    result = experiment_results[name]
    metrics = result["validation_metrics"]
    uncertainty = result["validation_uncertainty"]
    loss_rows.append({
        "experiment": result["display_name"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "mean_top_score": uncertainty["mean_confidence"],
        "score_interpretation": uncertainty["probability_interpretation"],
    })
display(pd.DataFrame(loss_rows).round(4))
for name in ("cnn_baseline", "bce_loss"):
    print("\n", experiment_results[name]["display_name"])
    display(pd.DataFrame(experiment_results[name]["validation_metrics"]["per_class"]).T.round(4))'''))

add(md(r"""## 9. ResNet18 feature extraction and fine-tuning

Both strategies retain standard `conv1` and `maxpool`, ImageNet normalization, and the same total training budget. Feature extraction freezes the backbone. Fine-tuning first trains the new head, then unfreezes `layer4` with a smaller learning rate. Parameter groups and every epoch's LRs are recorded."""))

add(code(r'''transfer_rows = []
for name in ("resnet18_feature_extraction", "resnet18_fine_tuning"):
    result = experiment_results[name]
    report = result["metadata"]["parameter_report_head_stage"]
    transfer_rows.append({
        "experiment": result["display_name"],
        "total_parameters": report["total_parameters"],
        "head_stage_trainable": report["trainable_parameters"],
        "head_stage_frozen": report["frozen_parameters"],
        "trainable_percent": report["trainable_percent"],
        "best_epoch": result["best_epoch"],
        "validation_macro_f1": result["validation_metrics"]["macro_f1"],
    })
display(pd.DataFrame(transfer_rows).round(4))

fine_history = pd.DataFrame([
    {
        "epoch": row["epoch"],
        "stage": row["stage"],
        "learning_rates": str(row["learning_rates"]),
        "trainable_parameters": str(row["trainable_parameters"]),
        "validation_macro_f1": row["validation_macro_f1"],
    }
    for row in experiment_results["resnet18_fine_tuning"]["history"]
])
display(fine_history)'''))

add(md(r"""## 10. Validation-only model selection and uncertainty

The production candidate with maximum validation macro-F1 is chosen before test. A scalar temperature is fit on validation logits, then the review threshold is the lowest confidence threshold reaching at least 80% automatic coverage. The resulting operational policy sends the remaining low-confidence volume to human review."""))

add(code(r'''uncertainty = json.loads((ROOT / "artifacts/results/validation_uncertainty.json").read_text(encoding="utf-8"))
final_result = json.loads((ROOT / "artifacts/results/frozen_test_evaluation.json").read_text(encoding="utf-8"))
print("Selected experiment:", final_result["selected_experiment"])
print("Validation-selected temperature:", uncertainty["temperature"])
print("Review threshold:", uncertainty["review_threshold"]["threshold"])
print("Target automatic coverage:", uncertainty["review_threshold"]["target_coverage"])
print("Validation calibration before:", uncertainty["uncalibrated"])
print("Validation calibration after:", uncertainty["calibrated"])
display(Image(filename=str(ROOT / "reports/figures/validation_risk_coverage.png"), width=700))'''))

add(md(r"""## 11. One frozen-test evaluation

The test set is now shown only because the selection record above is complete. It reports both confusion forms, all macro/per-class metrics, top mutual confusions, and low-confidence errors. It is not used to revise the model or threshold. The output archive, error table, selected checkpoint, and production publication are SHA-256-bound to this exact run."""))

add(code(r'''test_metrics = final_result["test_metrics"]
test_overall = pd.DataFrame([{
    "selected_model": final_result["selected_experiment"],
    "validation_macro_f1": final_result["selection_validation_metrics"]["macro_f1"],
    "test_accuracy": test_metrics["accuracy"],
    "test_macro_precision": test_metrics["macro_precision"],
    "test_macro_recall": test_metrics["macro_recall"],
    "test_macro_f1": test_metrics["macro_f1"],
    "test_ece": final_result["test_uncertainty"]["ece_10_bins"],
    "test_review_rate": final_result["test_selective"]["review_rate"],
}])
display(test_overall.round(4))
display(pd.DataFrame(test_metrics["per_class"]).T.round(4))
display(pd.DataFrame(final_result["mutual_confusion_pairs"]).head(10).round(4))'''))

add(code(r'''display(Image(filename=str(ROOT / "reports/figures/final_confusion_matrices.png"), width=1000))'''))
add(code(r'''display(Image(filename=str(ROOT / "reports/figures/misclassified_test_images.png"), width=1000))'''))
add(code(r'''test_errors = pd.read_csv(ROOT / "artifacts/results/test_errors.csv")
display(test_errors.head(12))'''))

add(md(r"""### Confusion and merge decision

The pair score is `C_norm[i,j] + C_norm[j,i]`. A large score triggers inspection, not an automatic merge. The final report assesses actual images, support, and whether the labels have distinct operational meaning before deciding whether to retain the eight-class taxonomy."""))

add(code(r'''top_pair = final_result["mutual_confusion_pairs"][0]
print("Strongest mutual pair:", top_pair)
print("Decision: retain the eight-class production taxonomy unless image inspection demonstrates redundant operational labels; confusion alone is insufficient.")'''))

add(md(r"""## 12. `unclean` and unseen `neysan`

The final model is applied to `unclean` only after removing all pixel-identical train/test copies. `neysan` is never treated as a ninth class. It is an unseen/human-review case. High known-class confidence is not proof against distribution shift."""))

add(code(r'''unclean_result = json.loads((ROOT / "artifacts/results/unclean_analysis.json").read_text(encoding="utf-8"))
unclean_predictions = pd.read_csv(ROOT / "artifacts/results/unclean_predictions.csv")
print("Cleaning:", unclean_result["cleaning"])
print("Known-class metrics:", unclean_result["known_class_metrics"])
print("Unseen neysan analysis:", unclean_result["unseen_neysan"])
display(unclean_predictions[unclean_predictions["is_unseen_neysan"]].sort_values("confidence").head(12))
display(Image(filename=str(ROOT / "reports/figures/unclean_low_confidence_examples.png"), width=1000))'''))

add(md(r"""## 13. Reproducible JSON prediction

The callable loads architecture, class order, transform metadata, temperature, threshold, seed, and weights from the final checkpoint. Before inference it verifies the run manifest, production-publication hash, selected source checkpoint, class mapping, transform, and frozen-test evidence hashes. Probabilities are Cross-Entropy softmax outputs."""))

add(code(r'''from traffic_classifier.predict import predict_image

sample_image = sorted((ROOT / "dataset/test/ambulance").glob("*.jpg"))[0]
prediction = predict_image(sample_image, checkpoint_path=ROOT / "artifacts/checkpoints/final_model.pt")
print("Image:", sample_image)
print(json.dumps(prediction, indent=2))
assert set(prediction["probabilities"]) == set(config.classes)
assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-3
assert isinstance(prediction["needs_review"], bool)'''))

add(md(r"""## 14. Conclusions and limitations

- The audit detects metadata-insensitive pixel duplicates across `unclean` and frozen splits, including a `vanet`/`neysan` label conflict.
- Controlled ablations use validation only; the combined regularized model and transfer strategy are interpreted with the small 80-image validation uncertainty in mind.
- Balanced batches improve fairness only when the operational objective values minority recall/macro-F1 rather than prevalence-weighted accuracy alone.
- Cross-entropy is retained for production because the classes are mutually exclusive and valid probabilities should sum to one.
- Human review mitigates uncertainty but does not make a tiny, cropped-image dataset error-free.
- Full class metrics, epoch histories, checkpoints metadata, errors, and transformed data evidence are saved rather than hidden in notebook state."""))

add(md(r"""## 15. Final checklist

- [x] Identical class mapping asserted for train and test.
- [x] Audit completed before validation creation and frozen test use.
- [x] Test used only once for final evaluation and its evidence archives are hash-bound.
- [x] A per-artifacts-directory OS lock prevents concurrent writers.
- [x] Cross-split duplicates detected with decoded image-content hashes.
- [x] Augmentation applied only to training.
- [x] Reproducible simulated imbalance and exact retained indices recorded.
- [x] Balanced batch has equal examples from all eight classes.
- [x] BCE uses one-hot float targets and argmax logits.
- [x] Scheduler uses validation behavior; per-epoch LRs are logged.
- [x] Per-epoch atomic checkpoints support automatic exact resume and completed checkpoints are reused.
- [x] Accuracy, precision, recall, F1, count/normalized confusions reported.
- [x] Per-class metrics and lowest classes compared across experiments.
- [x] At least 12 frozen-test errors plotted with labels/confidence.
- [x] `unclean` analyzed separately after duplicate removal.
- [x] `neysan` retained as unseen/review analysis, not a training class.
- [x] Final checkpoint contains mapping, transform, threshold, temperature, seed, and config.
- [x] JSON prediction and one-command reproduction are provided."""))

add(md("---"))
add(md(r"""# Appendix: Original Project Definition and Grading Rubric

The original assignment cells are retained unchanged below so the completed work remains directly traceable to every requirement."""))

for source in original_markdown:
    add(md(source))

notebook.cells = cells
notebook.metadata.update({
    "kernelspec": {
        "display_name": "Python 3.12 (Traffic Vehicle)",
        "language": "python",
        "name": "traffic-vehicle-python312",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12",
    },
    "project_completion": {
        "status": "complete",
        "pipeline": "python -m traffic_classifier.pipeline",
        "production_inference": "python predict.py IMAGE",
    },
})
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH} with {len(cells)} cells")
