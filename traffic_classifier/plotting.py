"""Publication-ready figures generated without relying on notebook-only state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def _figure_path(output_dir: str | Path, name: str) -> Path:
    destination = Path(output_dir) / "figures" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def plot_representative_images(
    paths: Sequence[str], class_names: Sequence[str], output_dir: str | Path
) -> Path:
    destination = _figure_path(output_dir, "representative_images.png")
    figure, axes = plt.subplots(4, 4, figsize=(14, 13))
    for axis, path, class_name in zip(axes.flat, paths, class_names):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(class_name, fontsize=10)
        axis.axis("off")
    figure.suptitle("Representative training images (two per class)", fontsize=15)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_size_distributions(frame: pd.DataFrame, output_dir: str | Path) -> Path:
    destination = _figure_path(output_dir, "image_size_distributions.png")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"train": "#2563eb", "test": "#16a34a", "unclean": "#ea580c"}
    for split, group in frame.groupby("split"):
        axes[0].hist(group["width"], bins=30, alpha=0.45, label=split, color=colors.get(split))
        axes[1].hist(group["height"], bins=30, alpha=0.45, label=split, color=colors.get(split))
        axes[2].scatter(
            group["width"], group["height"], s=9, alpha=0.35,
            label=split, color=colors.get(split)
        )
    axes[0].set(title="Image widths", xlabel="pixels", ylabel="count")
    axes[1].set(title="Image heights", xlabel="pixels", ylabel="count")
    axes[2].set(title="Width vs. height", xlabel="width", ylabel="height", xlim=(0, 600), ylim=(0, 1000))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_training_curves(
    histories: dict[str, list[dict[str, Any]]], output_dir: str | Path
) -> Path:
    destination = _figure_path(output_dir, "training_curves.png")
    selected = [
        name
        for name in (
            "cnn_baseline",
            "no_augmentation",
            "dropout_0_3",
            "dropout_0_5",
            "avg_pool",
            "weight_decay_1e-4",
            "step_lr",
            "bce_loss",
            "imbalanced_standard",
            "imbalanced_balanced",
            "best_regularized",
            "resnet18_feature_extraction",
            "resnet18_fine_tuning",
        )
        if name in histories
    ]
    if not selected:
        raise ValueError("No histories supplied")
    columns = min(3, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows * 2, columns, figsize=(5.2 * columns, 6.0 * rows), squeeze=False)
    for index, name in enumerate(selected):
        row = (index // columns) * 2
        column = index % columns
        history = histories[name]
        epochs = [item["epoch"] for item in history]
        axes[row, column].plot(
            epochs,
            [item["train_loss"] for item in history],
            label="train",
            color="#2563eb",
        )
        axes[row, column].plot(
            epochs,
            [item["validation_loss"] for item in history],
            label="validation",
            color="#dc2626",
        )
        axes[row, column].set(title=f"{name}: loss", xlabel="epoch", ylabel="loss")
        axes[row, column].grid(alpha=0.2)
        axes[row, column].legend(fontsize=8)
        axes[row + 1, column].plot(
            epochs,
            [item["validation_macro_f1"] for item in history],
            color="#7c3aed",
            marker="o",
            markersize=3,
        )
        axes[row + 1, column].set(
            title=f"{name}: validation macro-F1",
            xlabel="epoch",
            ylabel="macro-F1",
            ylim=(0, 1.02),
        )
        axes[row + 1, column].grid(alpha=0.2)
    for index in range(len(selected), rows * columns):
        row = (index // columns) * 2
        column = index % columns
        axes[row, column].axis("off")
        axes[row + 1, column].axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_confusion_matrices(
    metrics: dict[str, Any], class_names: Sequence[str], output_dir: str | Path
) -> Path:
    destination = _figure_path(output_dir, "final_confusion_matrices.png")
    counts = np.asarray(metrics["confusion_matrix_counts"], dtype=float)
    normalized = np.asarray(metrics["confusion_matrix_row_normalized"], dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    images = [
        (counts, "Counts", "Blues", None),
        (normalized, "Row-normalized", "Oranges", "%.2f"),
    ]
    for axis, (matrix, title, color, formatting) in zip(axes, images):
        draw = axis.imshow(matrix, cmap=color, vmin=0 if formatting is None else None, vmax=1 if formatting else None)
        axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), labels=class_names)
        axis.set(xlabel="Predicted class", ylabel="True class", title=f"Confusion matrix — {title}")
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
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_per_class_comparison(
    experiment_metrics: dict[str, dict[str, Any]],
    class_names: Sequence[str],
    output_dir: str | Path,
) -> Path:
    destination = _figure_path(output_dir, "per_class_comparison.png")
    names = list(experiment_metrics)
    if not names:
        raise ValueError("No experiment metrics supplied")
    figure, axes = plt.subplots(1, 3, figsize=(19, 6), sharex=True)
    for axis, metric_name in zip(axes, ("precision", "recall", "f1")):
        matrix = np.asarray(
            [
                [experiment_metrics[name]["per_class"][class_name][metric_name] for class_name in class_names]
                for name in names
            ]
        )
        draw = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(names)), labels=names)
        axis.set_title(metric_name.capitalize())
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[row, column] < 0.55 else "black")
        figure.colorbar(draw, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Per-class validation metrics for the main experiments")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_error_gallery(
    paths: Sequence[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    count: int = 12,
) -> Path:
    destination = _figure_path(output_dir, "misclassified_test_images.png")
    incorrect = np.flatnonzero(targets != predictions)
    order = incorrect[np.argsort(probabilities[incorrect].max(axis=1), kind="stable")]
    chosen = order[: min(count, len(order))]
    columns = 4
    rows = max(1, int(np.ceil(len(chosen) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 4.0 * rows), squeeze=False)
    for axis, index in zip(axes.flat, chosen):
        with Image.open(paths[index]) as image:
            axis.imshow(image.convert("RGB"))
        confidence = float(probabilities[index].max())
        axis.set_title(
            f"true: {class_names[targets[index]]}\npred: {class_names[predictions[index]]}\nconfidence: {confidence:.3f}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in list(axes.flat)[len(chosen) :]:
        axis.axis("off")
    if not len(chosen):
        axes[0, 0].text(0.5, 0.5, "No frozen-test errors", ha="center", va="center")
        axes[0, 0].axis("off")
    figure.suptitle(
        f"Lowest-confidence frozen-test misclassifications ({len(chosen)} shown)",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_review_coverage(
    curve: Sequence[dict[str, float]], output_dir: str | Path
) -> Path:
    destination = _figure_path(output_dir, "validation_risk_coverage.png")
    coverages = [row["coverage"] for row in curve]
    risks = [row["selective_risk"] for row in curve]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(coverages, risks, color="#7c3aed", linewidth=2)
    axis.set(
        xlabel="Automatic coverage",
        ylabel="Error risk among automatically accepted images",
        title="Validation risk–coverage curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_unclean_analysis(
    paths: Sequence[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    unseen_index: int,
    unseen_name: str = "neysan",
) -> Path:
    destination = _figure_path(output_dir, "unclean_low_confidence_examples.png")
    neysan = np.flatnonzero(targets == unseen_index)
    order = neysan[np.argsort(probabilities[neysan].max(axis=1), kind="stable")][:12]
    figure, axes = plt.subplots(3, 4, figsize=(15, 11), squeeze=False)
    for axis, index in zip(axes.flat, order):
        with Image.open(paths[index]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"unseen: {unseen_name}\npred: {class_names[predictions[index]]}\nconfidence: {probabilities[index].max():.3f}",
            fontsize=9,
        )
        axis.axis("off")
    figure.suptitle(f"Lowest-confidence unseen `{unseen_name}` examples", fontsize=15)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination
