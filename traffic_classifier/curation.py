"""Export the confident known-class subset of a scored split into a clean folder.

Selection is deliberately conservative: an image is kept only when its source
label is a real training class, the model agreed with that label, and the
calibrated confidence clears the validation-derived review threshold. The
unseen `neysan` class is never promoted to a training label here.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .utils import ensure_directories

MANIFEST_COLUMNS = (
    "source_path",
    "destination_path",
    "class_name",
    "confidence",
    "status",
)


def select_confident_known_rows(
    frame: pd.DataFrame,
    classes: Sequence[str],
    review_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a prediction table into kept rows and rows rejected with a reason.

    Rejections are exclusive and evaluated in order: unseen class, then
    disagreement with the source label, then insufficient confidence.
    """

    required = {
        "path",
        "source_label",
        "predicted_class",
        "confidence",
        "is_unseen_neysan",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")

    known = set(classes)
    reasons: list[str] = []
    for row in frame.itertuples(index=False):
        if bool(row.is_unseen_neysan):
            reasons.append("unseen_class")
        elif row.source_label not in known:
            reasons.append("label_not_a_known_class")
        elif row.source_label != row.predicted_class:
            reasons.append("model_disagrees_with_label")
        elif float(row.confidence) < review_threshold:
            reasons.append("confidence_below_review_threshold")
        else:
            reasons.append("kept")
    labelled = frame.assign(selection_reason=reasons)
    kept = labelled[labelled["selection_reason"] == "kept"].reset_index(drop=True)
    rejected = labelled[labelled["selection_reason"] != "kept"].reset_index(drop=True)
    return kept, rejected


def _source_path(row: Any, source_root: Path) -> Path:
    """Resolve a recorded path, falling back to the split-root class folder."""

    recorded = Path(str(row.path))
    if recorded.is_absolute() and recorded.exists():
        return recorded
    candidate = source_root / str(row.source_label) / recorded.name
    return candidate if candidate.exists() else recorded


def _plan_destinations(
    kept: pd.DataFrame, source_root: Path, destination_root: Path
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in kept.itertuples(index=False):
        source = _source_path(row, source_root)
        destination = destination_root / str(row.source_label) / source.name
        class_name = str(row.source_label)
        if not source.exists():
            status = "missing_source"
        elif str(destination) in seen or destination.exists():
            status = "destination_exists"
        else:
            status = "planned"
            seen.add(str(destination))
        plan.append(
            {
                "source_path": source.as_posix(),
                "destination_path": destination.as_posix(),
                "class_name": class_name,
                "confidence": float(row.confidence),
                "status": status,
            }
        )
    return plan


def export_confident_subset(
    frame: pd.DataFrame,
    classes: Sequence[str],
    review_threshold: float,
    source_root: str | Path,
    destination_root: str | Path,
    *,
    mode: str = "copy",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Materialize the confident subset as an ImageFolder-compatible directory.

    `mode` is `copy` to leave the source split untouched, or `move` to relocate
    the files. In both cases the manifest is returned before anything is
    written to disk so callers can persist the intended plan first.
    """

    if mode not in {"copy", "move"}:
        raise ValueError("mode must be 'copy' or 'move'")

    source_root = Path(source_root)
    destination_root = Path(destination_root)
    kept, rejected = select_confident_known_rows(frame, classes, review_threshold)
    plan = _plan_destinations(kept, source_root, destination_root)

    transferable = [item for item in plan if item["status"] == "planned"]
    if transferable:
        for class_name in sorted({item["class_name"] for item in transferable}):
            ensure_directories(destination_root / class_name)
        for item in transferable:
            source = Path(item["source_path"])
            destination = Path(item["destination_path"])
            if mode == "move":
                shutil.move(str(source), str(destination))
            else:
                shutil.copy2(source, destination)
            item["status"] = "moved" if mode == "move" else "copied"

    manifest = pd.DataFrame(plan, columns=list(MANIFEST_COLUMNS))
    summary = {
        "mode": mode,
        "source_root": source_root.as_posix(),
        "destination_root": destination_root.as_posix(),
        "review_threshold": float(review_threshold),
        "selection_rule": (
            "source label is a known class, model prediction matches it, and "
            "confidence is at or above the validation review threshold"
        ),
        "rows_considered": int(len(frame)),
        "rows_selected": int(len(kept)),
        "rows_transferred": int(
            (manifest["status"].isin(["moved", "copied"])).sum()
        ),
        "rejection_counts": (
            rejected["selection_reason"].value_counts().sort_index().to_dict()
            if len(rejected)
            else {}
        ),
        "skipped": (
            manifest[~manifest["status"].isin(["planned", "moved", "copied"])][
                "status"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "class_counts": (
            kept["source_label"].value_counts().sort_index().to_dict()
            if len(kept)
            else {}
        ),
    }
    return manifest, rejected, summary


def verify_export(
    manifest: pd.DataFrame, destination_root: str | Path
) -> dict[str, Any]:
    """Check that every transferred file exists at its destination."""

    destination_root = Path(destination_root)
    transferred = manifest[manifest["status"].isin(["moved", "copied"])]
    missing: list[str] = []
    for item in transferred.itertuples(index=False):
        destination = Path(item.destination_path)
        if not destination.exists():
            missing.append(destination.as_posix())
    return {
        "transferred": int(len(transferred)),
        "missing_at_destination": missing,
        "verified": not missing,
    }