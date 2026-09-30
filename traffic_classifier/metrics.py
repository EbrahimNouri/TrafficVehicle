"""Classification, calibration, uncertainty, and confusion-pair metrics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


ProbabilityKind = Literal["softmax", "sigmoid", "none"]


def probabilities_from_logits(
    logits: torch.Tensor | np.ndarray,
    kind: ProbabilityKind = "softmax",
    temperature: float = 1.0,
) -> np.ndarray:
    values = torch.as_tensor(logits, dtype=torch.float32)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = values / temperature
    if kind == "softmax":
        return torch.softmax(values, dim=1).cpu().numpy()
    if kind == "sigmoid":
        return torch.sigmoid(values).cpu().numpy()
    if kind == "none":
        return values.cpu().numpy()
    raise ValueError(f"Unknown probability kind: {kind}")


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    row_normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "per_class": per_class,
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_row_normalized": row_normalized.tolist(),
    }


def multiclass_nll(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    selected = clipped[np.arange(len(y_true)), y_true]
    return float(-np.log(selected).mean())


def expected_calibration_error(
    probabilities: np.ndarray, y_true: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == 0:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit a scalar temperature by deterministic NLL grid search."""

    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in np.linspace(0.5, 3.0, 501):
        probabilities = probabilities_from_logits(
            logits, kind="softmax", temperature=float(temperature)
        )
        nll = multiclass_nll(probabilities, y_true)
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
    return best_temperature


def select_review_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    target_coverage: float = 0.80,
) -> dict[str, Any]:
    """Choose the lowest threshold meeting a validation coverage target.

    Images below the threshold are sent to human review. Targeting 80% automatic
    coverage creates a bounded 20% review burden while selecting a threshold from
    validation data rather than the frozen test set.
    """

    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    candidates = np.unique(confidence)[::-1]
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    total = len(confidence)
    for threshold in candidates:
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        selective_accuracy = float(correct[accepted].mean()) if accepted.any() else 0.0
        risk = 1.0 - selective_accuracy
        row = {
            "threshold": float(threshold),
            "coverage": coverage,
            "review_rate": 1.0 - coverage,
            "selective_accuracy": selective_accuracy,
            "selective_risk": risk,
        }
        rows.append(row)
        if coverage >= target_coverage and selected is None:
            selected = row
    if selected is None:  # target coverage can only fail under tied confidences
        accepted = confidence >= float(candidates[0])
        selected = {
            "threshold": float(candidates[0]),
            "coverage": float(accepted.mean()),
            "review_rate": float(1.0 - accepted.mean()),
            "selective_accuracy": float(correct[accepted].mean()),
            "selective_risk": float(1.0 - correct[accepted].mean()),
        }
    return {
        "selection_rule": "lowest validation threshold meeting target automatic coverage",
        "target_coverage": target_coverage,
        **selected,
        "accepted_count": int(round(selected["coverage"] * total)),
        "review_count": int(round(selected["review_rate"] * total)),
        "validation_count": total,
        "curve": rows,
    }


def risk_coverage_curve(
    probabilities: np.ndarray, y_true: np.ndarray, points: int = 101
) -> list[dict[str, float]]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    order = np.argsort(-confidence, kind="stable")
    cumulative_correct = np.cumsum(correct[order])
    coverages = np.linspace(0.0, 1.0, points)
    rows: list[dict[str, float]] = []
    for coverage in coverages:
        count = int(round(coverage * len(confidence)))
        if count == 0:
            selective_accuracy = 1.0 - float(1.0 - correct.mean())
            risk = float(1.0 - correct.mean())
        else:
            risk = float(1.0 - cumulative_correct[count - 1] / count)
            selective_accuracy = 1.0 - risk
        rows.append(
            {
                "coverage": float(coverage),
                "selective_risk": risk,
                "selective_accuracy": float(selective_accuracy),
            }
        )
    return rows


def confusion_pair_ranking(
    normalized_confusion: np.ndarray | list[list[float]],
    class_names: list[str],
) -> list[dict[str, Any]]:
    matrix = np.asarray(normalized_confusion, dtype=float)
    pairs: list[dict[str, Any]] = []
    for first in range(len(class_names)):
        for second in range(first + 1, len(class_names)):
            pairs.append(
                {
                    "class_a": class_names[first],
                    "class_b": class_names[second],
                    "a_as_b": float(matrix[first, second]),
                    "b_as_a": float(matrix[second, first]),
                    "pair_confusion": float(matrix[first, second] + matrix[second, first]),
                }
            )
    return sorted(pairs, key=lambda row: row["pair_confusion"], reverse=True)


def worst_classes(
    metrics: dict[str, Any], metric_name: str
) -> tuple[str, float]:
    per_class = metrics["per_class"]
    name = min(per_class, key=lambda key: per_class[key][metric_name])
    return name, float(per_class[name][metric_name])
