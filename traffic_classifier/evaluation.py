"""Reusable evaluation of a classification model on an image-folder split.

This is the importable counterpart of ``scripts/evaluate_all_models_on_test.py``
(which orchestrates the all-models diagnostic over ``dataset/test`` by looping
this building block). Use it to evaluate a single model or any split:

.. code-block:: python

    from traffic_classifier.evaluation import Evaluation, build_evaluation_model

    checkpoint = load_checkpoint(path, map_location="cpu")
    model, info = build_evaluation_model(checkpoint, class_names, device)
    evaluation = Evaluation(
        name=info["name"],
        display_name=info["display_name"],
        architecture=info["architecture"],
        loss=info["loss"],
        model=model,
        class_names=class_names,
        device=device,
        probability_kind=probability_kind_for_loss(info["loss"]),
    )
    result = evaluation.run(loader, temperature=temperature)
    rows = evaluation.per_image_rows(sample_paths, temperature, review_threshold)

CLI (evaluate one model on ``dataset/test``, e.g. the production model):

.. code-block:: bash

    python -m traffic_classifier.evaluation
    python -m traffic_classifier.evaluation \
        --checkpoint artifacts/checkpoints/resnet18_fine_tuning.pt \
        --output-dir artifacts/results/evaluation_resnet18_fine_tuning

Like the all-models diagnostic, this is read-only with respect to model
selection: it never writes selection artifacts or the official frozen-test
evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .data import build_transforms, load_image_folder, make_loader
from .engine import evaluate
from .metrics import (
    classification_metrics,
    expected_calibration_error,
    multiclass_nll,
)
from .models import build_model
from .predict import load_production_model
from .utils import load_checkpoint

PER_IMAGE_COLUMNS = [
    "model",
    "display_name",
    "architecture",
    "loss",
    "probability_kind",
    "image_index",
    "image_path",
    "true_class",
    "predicted_class",
    "correct",
    "confidence",
    "margin",
    "true_class_probability",
    "top3",
    "temperature",
    "below_official_review_threshold",
]


def probability_kind_for_loss(loss: str | None) -> str:
    return "sigmoid" if loss == "bce" else "softmax"


def build_evaluation_model(
    checkpoint: dict[str, Any],
    class_names: Sequence[str],
    device: torch.device,
    *,
    entry: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model from a checkpoint (optionally guided by an experiment record).

    ``entry`` is a record from ``artifacts/results/experiment_results.json``;
    when omitted the checkpoint's own ``train_options`` / ``metadata`` are used,
    which is what the standalone checkpoints under ``artifacts/checkpoints``
    carry.
    """

    metadata = dict(checkpoint.get("metadata") or {})
    options = dict(checkpoint.get("train_options") or {})
    if entry is not None:
        metadata = {**metadata, **dict(entry.get("metadata") or {})}
        architecture = str(entry.get("architecture") or options.get("architecture") or "cnn")
        loss = entry.get("loss")
        display_name = entry.get("display_name") or metadata.get("display_name", "experiment")
    else:
        architecture = str(options.get("architecture") or metadata.get("architecture") or "cnn")
        loss = options.get("loss")
        display_name = metadata.get("display_name") or options.get("experiment_name") or "model"

    model = build_model(
        architecture,
        len(class_names),
        dropout=float(metadata.get("dropout", 0.0)),
        pooling=str(metadata.get("pooling", "max")),  # type: ignore[arg-type]
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    info = {
        "name": options.get("experiment_name") or metadata.get("name") or display_name,
        "display_name": display_name,
        "architecture": architecture,
        "loss": loss,
        "temperature": float(checkpoint.get("temperature", 1.0)),
        "threshold": float(checkpoint.get("threshold", 0.70)),
    }
    return model, info


def confidence_summary(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Confidence, margin, ECE and NLL over a batch of predictions."""

    probs = np.asarray(probabilities)
    predictions = probs.argmax(axis=1)
    confidence = probs[np.arange(len(predictions)), predictions]
    if probs.shape[1] > 1:
        runner_up = np.sort(probs, axis=1)[:, -2]
        margin = confidence - runner_up
    else:
        margin = confidence
    return {
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "mean_margin": float(margin.mean()),
        "ece_argmax_10_bins": expected_calibration_error(probs, targets),
        "nll": multiclass_nll(probs, targets),
        "correct": int((predictions == targets).sum()),
    }


def confidence_by_class(
    rows: list[dict[str, Any]], classes: Sequence[str]
) -> list[dict[str, Any]]:
    """Aggregate confidence and accuracy per true class, per model."""

    summary: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        for label in classes:
            subset = [row for row in model_rows if row["true_class"] == label]
            if not subset:
                continue
            confidences = [row["confidence"] for row in subset]
            correct = [row["confidence"] for row in subset if row["correct"]]
            wrong = [row["confidence"] for row in subset if not row["correct"]]
            summary.append(
                {
                    "model": model,
                    "true_class": label,
                    "images": len(subset),
                    "accuracy": len(correct) / len(subset),
                    "mean_confidence": float(np.mean(confidences)),
                    "median_confidence": float(np.median(confidences)),
                    "min_confidence": float(min(confidences)),
                    "max_confidence": float(max(confidences)),
                    "mean_confidence_when_correct": (
                        float(np.mean(correct)) if correct else ""
                    ),
                    "mean_confidence_when_wrong": (
                        float(np.mean(wrong)) if wrong else ""
                    ),
                }
            )
    return summary


class Evaluation:
    """Run one trained model over a split and report per-image and aggregate scores.

    The model instance is built by :func:`build_evaluation_model` and must
    already be in eval mode. ``run()`` executes the model over a
    ``make_loader`` DataLoader and keeps the predictions for
    :meth:`per_image_rows` and :meth:`write_outputs`.
    """

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        architecture: str,
        loss: str | None,
        model: nn.Module,
        class_names: Sequence[str],
        device: torch.device,
        probability_kind: str = "softmax",
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.architecture = architecture
        self.loss = loss or "cross_entropy"
        self.model = model
        self.class_names = list(class_names)
        self.device = device
        self.probability_kind = probability_kind
        self._criterion = (
            nn.BCEWithLogitsLoss() if self.loss == "bce" else nn.CrossEntropyLoss()
        )
        self._outputs: dict[str, np.ndarray] | None = None
        self._last_rows: list[dict[str, Any]] = []

    @torch.inference_mode()
    def run(self, loader: DataLoader[Any], *, temperature: float = 1.0) -> dict[str, Any]:
        """Evaluate and return classification metrics plus per-image outputs."""

        result = evaluate(
            self.model,
            loader,
            self._criterion,
            self.device,
            loss_name=self.loss,
            num_classes=len(self.class_names),
            class_names=self.class_names,
            probability_kind=self.probability_kind,
            temperature=temperature,
        )
        self._outputs = result["outputs"]
        result["uncertainty"] = confidence_summary(
            result["outputs"]["probabilities"], result["outputs"]["targets"]
        )
        return result

    @property
    def outputs(self) -> dict[str, np.ndarray]:
        if self._outputs is None:
            raise RuntimeError("run() must be called before accessing outputs")
        return self._outputs

    def per_image_rows(
        self,
        sample_paths: Sequence[str],
        *,
        temperature: float = 1.0,
        review_threshold: float,
    ) -> list[dict[str, Any]]:
        """One row per image: chosen-class confidence, margin, top-3, review flag."""

        probabilities = self.outputs["probabilities"]
        targets = self.outputs["targets"]
        classes = self.class_names
        rows: list[dict[str, Any]] = []
        for index in range(len(targets)):
            scores = probabilities[index]
            order = np.argsort(-scores)
            top3 = "|".join(
                f"{classes[position]}:{scores[position]:.6f}" for position in order[:3]
            )
            confidence = float(scores[order[0]])
            runner_up = float(scores[order[1]]) if len(order) > 1 else 0.0
            true_index = int(targets[index])
            rows.append(
                {
                    "model": self.name,
                    "display_name": self.display_name,
                    "architecture": self.architecture,
                    "loss": self.loss,
                    "probability_kind": self.probability_kind,
                    "image_index": index,
                    "image_path": sample_paths[index],
                    "true_class": classes[true_index],
                    "predicted_class": classes[int(order[0])],
                    "correct": bool(order[0] == true_index),
                    "confidence": confidence,
                    "margin": confidence - runner_up,
                    "true_class_probability": float(scores[true_index]),
                    "top3": top3,
                    "temperature": temperature,
                    "below_official_review_threshold": bool(
                        confidence < review_threshold
                    ),
                }
            )
        return rows

    def write_outputs(self, output_dir: Path) -> dict[str, Path]:
        """Write per-image CSV, metrics JSON and a confusion-matrix PNG."""

        output_dir.mkdir(parents=True, exist_ok=True)
        if self._outputs is None:
            raise RuntimeError("run() must be called before write_outputs()")
        metrics = classification_metrics(
            np.asarray(self._outputs["targets"]),
            np.asarray(self._outputs["predictions"]),
            self.class_names,
        )
        csv_path = output_dir / f"{self.name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PER_IMAGE_COLUMNS)
            writer.writeheader()
            writer.writerows(self._last_rows)
        metrics_path = output_dir / f"{self.name}_metrics.json"
        payload = {
            **metrics,
            "uncertainty": confidence_summary(
                self._outputs["probabilities"], self._outputs["targets"]
            ),
        }
        metrics_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        png_path = output_dir / f"{self.name}_confusion.png"
        write_confusion_png(metrics, self.class_names, png_path)
        return {"csv": csv_path, "metrics": metrics_path, "confusion": png_path}

    def _attach_rows(self, rows: list[dict[str, Any]]) -> None:
        self._last_rows = rows


def write_confusion_png(
    metrics: dict[str, Any],
    class_names: Sequence[str],
    path: Path,
) -> Path:
    """Row-normalized confusion heatmap with raw count annotations."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(metrics["confusion_matrix_counts"], dtype=float)
    normalized = np.asarray(metrics["confusion_matrix_row_normalized"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for axis, (matrix, title, color, formatting) in zip(
        axes,
        [
            (counts, "Counts", "Blues", None),
            (normalized, "Row-normalized", "Oranges", "%.2f"),
        ],
    ):
        draw = axis.imshow(
            matrix, cmap=color, vmin=0 if formatting is None else None, vmax=1 if formatting else None
        )
        axis.set_xticks(range(len(class_names)), labels=list(class_names), rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), labels=list(class_names))
        axis.set(xlabel="Predicted class", ylabel="True class", title=f"Confusion matrix - {title}")
        threshold = matrix.max() / 2.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.0f}" if formatting is None else f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )
        figure.colorbar(draw, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one trained model over an image-folder split "
            "(default dataset/test) and optionally write per-image CSV, "
            "metrics JSON and a confusion PNG."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/final_model.pt",
        help="Checkpoint path (production final_model.pt or any experiment .pt)",
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="When given, write <name>.csv, <name>_metrics.json, <name>_confusion.png",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = ProjectConfig.from_json(root / args.config)
    device = torch.device(args.device) if args.device else config.resolve_device()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path

    is_production = checkpoint_path.name == "final_model.pt" and (
        checkpoint_path.parent.parent / "production_ready.json"
    ).is_file()
    if is_production:
        model, checkpoint, config, device = load_production_model(checkpoint_path, device)
        options = checkpoint.get("train_options") or {}
        architecture = options.get("architecture") or "cnn"
        loss = options.get("loss") or "cross_entropy"
        class_names = list(checkpoint.get("class_names") or config.classes)
        display_name = checkpoint.get("metadata", {}).get("display_name") or "production model"
        name = checkpoint.get("metadata", {}).get("name") or options.get("experiment_name") or "final_model"
        temperature = float(checkpoint.get("temperature", 1.0))
        threshold = float(checkpoint.get("threshold", 0.70))
        transform_kind = (checkpoint.get("transform") or {}).get(
            "name", "resnet" if architecture == "resnet18" else "cnn"
        )
    else:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        model, info = build_evaluation_model(checkpoint, config.classes, device)
        architecture = info["architecture"]
        loss = info["loss"]
        class_names = list(checkpoint.get("class_names") or config.classes)
        display_name = info["display_name"]
        name = info["name"]
        temperature = info["temperature"]
        threshold = info["threshold"]
        metadata = checkpoint.get("metadata") or {}
        transform_kind = (metadata.get("evaluation_transform") or {}).get(
            "name", "resnet" if architecture == "resnet18" else "cnn"
        )

    transform, _ = build_transforms(
        config, augmentation=False, kind=transform_kind  # type: ignore[arg-type]
    )
    dataset = load_image_folder(
        config.data_dir,
        args.split,
        transform,
        class_names,
        remove_empty_classes=False,
        allow_empty=True,
    )
    loader = make_loader(
        dataset,
        range(len(dataset)),
        training=False,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    evaluation = Evaluation(
        name=name,
        display_name=display_name,
        architecture=architecture,
        loss=loss,
        model=model,
        class_names=class_names,
        device=device,
        probability_kind=probability_kind_for_loss(loss),
    )
    result = evaluation.run(loader, temperature=temperature)
    paths_list = [Path(sample[0]).as_posix() for sample in dataset.samples]
    rows = evaluation.per_image_rows(
        paths_list, temperature=temperature, review_threshold=threshold
    )
    evaluation._attach_rows(rows)

    metrics = result
    print(f"checkpoint : {checkpoint_path.as_posix()}")
    print(f"split      : {config.data_dir}/{args.split} ({len(rows)} images)")
    print(f"accuracy   : {metrics['accuracy']:.4f}")
    print(f"macro-F1   : {metrics['macro_f1']:.4f}")
    print(f"macro-P    : {metrics['macro_precision']:.4f}")
    print(f"macro-R    : {metrics['macro_recall']:.4f}")
    print(f"weighted F1: {metrics['weighted_f1']:.4f}")
    uncertainty = metrics["uncertainty"]
    print(
        f"confidence : mean {uncertainty['mean_confidence']:.4f} "
        f"margin {uncertainty['mean_margin']:.4f} "
        f"ECE {uncertainty['ece_argmax_10_bins']:.4f} "
        f"NLL {uncertainty['nll']:.4f}"
    )
    below = sum(1 for row in rows if row["below_official_review_threshold"])
    print(f"below threshold {threshold:.4f}: {below}/{len(rows)}")

    print("\nper-class:")
    print(f"{'class':<12}{'n':>5} {'P':>7} {'R':>7} {'F1':>7}")
    for class_name in class_names:
        cell = metrics["per_class"][class_name]
        print(
            f"{class_name:<12}{cell['support']:>5} "
            f"{cell['precision']:>7.3f} {cell['recall']:>7.3f} {cell['f1']:>7.3f}"
        )
    if args.output_dir:
        written = evaluation.write_outputs(Path(args.output_dir))
        for kind, path in written.items():
            print(f"written {kind:<9}: {path}")


if __name__ == "__main__":
    main()