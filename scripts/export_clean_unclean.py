"""Export the confident known-class subset of `unclean` into a new split.

Reads the prediction table produced by the pipeline and materializes only the
images whose class is certain: a known training label that the model agrees
with, at or above the calibrated review threshold. The unseen `neysan` class is
never promoted to a training label.

Examples:
    Python scripts/export_clean_unclean.py --dry-run
    Python scripts/export_clean_unclean.py --mode copy
    Python scripts/export_clean_unclean.py --mode move --force
    Python scripts/export_clean_unclean.py --mode move --verify-then-move --force
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


# Resolve the project root (the parent of the scripts/ directory) and make it
# available on sys.path so the package imports work regardless of the working
# directory the script is invoked from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_classifier.config import ProjectConfig  # noqa: E402
from traffic_classifier.curation import (  # noqa: E402
    MANIFEST_COLUMNS,
    export_confident_subset,
    select_confident_known_rows,
    verify_export,
)
from traffic_classifier.data import count_images  # noqa: E402
from traffic_classifier.utils import (  # noqa: E402
    atomic_write_csv,
    read_json,
    write_json,
)


def _resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def _review_threshold(
    uncertainty_path: Path, override: float | None
) -> tuple[float, str]:
    if override is not None:
        return float(override), "command line override"
    payload = read_json(uncertainty_path)
    threshold = float(payload["review_threshold"]["threshold"])
    return threshold, uncertainty_path.as_posix()


def _preflight_verify(
    frame: pd.DataFrame,
    classes: Sequence[str],
    threshold: float,
    source_root: Path,
    destination_root: Path,
) -> tuple[bool, list[str]]:
    """Check every selected source file before any destructive operation.

    Verifies that the selection is non-empty, the source split exists, the
    destination parent is writable, and every selected source file is present
    and readable. Returns ``(ok, problems)``; if ``problems`` is non-empty the
    caller must abort and leave the filesystem untouched.
    """

    problems: list[str] = []

    kept, _ = select_confident_known_rows(frame, classes, threshold)
    if kept.empty:
        problems.append("pre-flight: selection is empty; nothing to move")
        return False, problems

    if not source_root.is_dir():
        problems.append(f"pre-flight: source split does not exist: {source_root}")
        return False, problems

    # Destination parent must exist and be writable before we touch any file.
    parent = destination_root.parent
    if not parent.exists():
        problems.append(f"pre-flight: destination parent missing: {parent}")
        return False, problems
    if not os.access(parent, os.W_OK):
        problems.append(f"pre-flight: destination parent not writable: {parent}")
        return False, problems

    missing: list[str] = []
    unreadable: list[str] = []
    for _, row in kept.iterrows():
        src = Path(row["absolute_path"])
        if not src.is_file():
            missing.append(str(src))
            continue
        try:
            with src.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            unreadable.append(f"{src}: {exc}")

    if missing:
        problems.append(
            f"pre-flight: {len(missing)} selected file(s) missing on disk "
            f"(first: {missing[0]})"
        )
    if unreadable:
        problems.append(
            f"pre-flight: {len(unreadable)} selected file(s) unreadable "
            f"(first: {unreadable[0]})"
        )

    if not problems:
        print(
            f"Pre-flight: {len(kept)} selected file(s) present and readable; "
            f"destination parent writable."
        )
    return not problems, problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export confident known-class images from a scored split"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.json"),
        help="Path to the project config JSON (default: <project>/configs/default.json)",
    )
    parser.add_argument(
        "--predictions",
        default=str(ROOT / "artifacts" / "results" / "unclean_predictions.csv"),
        help="Prediction table produced by the pipeline (default under <project>/artifacts/results)",
    )
    parser.add_argument(
        "--uncertainty",
        default=str(ROOT / "artifacts" / "results" / "validation_uncertainty.json"),
        help="Validation uncertainty artifact used to read the review threshold",
    )
    parser.add_argument("--source-split", default="unclean")
    parser.add_argument("--destination-split", default="cleaned_unclean")
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="copy keeps the source split intact; move relocates the files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the review threshold from the validation artifact",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the selection and planned transfers without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required for --mode move, which removes images from the source split",
    )
    parser.add_argument(
        "--verify-then-move",
        action="store_true",
        help=(
            "Run a pre-flight verification of every selected source file "
            "(existence, readability, destination writability) and only then "
            "perform the move. If any check fails, abort before touching any "
            "file. Only valid together with --mode move."
        ),
    )
    parser.add_argument(
        "--allow-missing-source",
        action="store_true",
        help=(
            "Deprecated and ignored: a source split that is missing or holds no "
            "images is now always skipped cleanly (status 0) instead of raising."
        ),
    )
    args = parser.parse_args()

    if args.allow_missing_source:
        print("[note] --allow-missing-source is deprecated and has no effect.")

    if args.mode == "move" and not args.force and not args.dry_run:
        parser.error("--mode move requires --force because it empties the source split")

    if args.verify_then_move and args.dry_run:
        parser.error("--verify-then-move and --dry-run are mutually exclusive")

    if args.verify_then_move and args.mode != "move":
        parser.error("--verify-then-move only makes sense with --mode move")

    config_path = _resolve(args.config)
    predictions_path = _resolve(args.predictions)
    uncertainty_path = _resolve(args.uncertainty)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            "Run the pipeline first or pass --config explicitly."
        )

    config = ProjectConfig.from_json(config_path)
    source_root = Path(config.data_dir) / args.source_split
    destination_root = Path(config.data_dir) / args.destination_split
    results_dir = Path(config.artifacts_dir) / "results"
    # If config.data_dir is relative, resolve it against the project root so the
    # script behaves the same regardless of where it is executed from.
    if not Path(config.data_dir).is_absolute():
        source_root = (ROOT / source_root).resolve()
        destination_root = (ROOT / destination_root).resolve()

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction table not found: {predictions_path}. Run the pipeline first."
        )
    if not uncertainty_path.exists():
        raise FileNotFoundError(
            f"Uncertainty artifact not found: {uncertainty_path}. Run the pipeline first."
        )

    frame = pd.read_csv(predictions_path)
    if frame.empty:
        print(
            f"Prediction table {predictions_path} is empty; nothing to export."
        )
        return

    threshold, threshold_source = _review_threshold(uncertainty_path, args.threshold)
    print(f"Review threshold: {threshold:.6f} (from {threshold_source})")

    # Pre-flight verification: only run when explicitly requested. Default
    # behavior (no flag) is unchanged so existing callers keep working.
    if args.verify_then_move:
        print("Running pre-flight verification before move...")
        ok, problems = _preflight_verify(
            frame, config.classes, threshold, source_root, destination_root
        )
        if not ok:
            print("PRE-FLIGHT FAILED. No files were moved.")
            for problem in problems:
                print(f"  - {problem}")
            sys.exit(2)
        print("Pre-flight OK. Proceeding with move.")

    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "cleaned_unclean_manifest.csv"
    summary_path = results_dir / "cleaned_unclean_summary.json"
    rejected_path = results_dir / "cleaned_unclean_rejected.csv"

    # A source split can exist and still hold no image: a prior `--mode move`
    # export drains it completely but leaves the class folders behind. Counting
    # images (not directory entries) is what distinguishes that from a populated
    # split; there is nothing to export, so skip instead of raising.
    source_images = count_images(source_root)
    if not source_images:
        skip_reason = (
            f"Source split {source_root.as_posix()} holds no image files; "
            "nothing to export."
        )
        print(skip_reason)
        empty_manifest = pd.DataFrame(columns=list(MANIFEST_COLUMNS))
        empty_rejected = pd.DataFrame(
            columns=[*frame.columns, "selection_reason"]
        )
        atomic_write_csv(empty_manifest, manifest_path)
        atomic_write_csv(empty_rejected, rejected_path)
        summary = {
            "mode": args.mode,
            "source_root": source_root.as_posix(),
            "destination_root": destination_root.as_posix(),
            "review_threshold": float(threshold),
            "selection_rule": "not evaluated: the source split holds no images",
            "rows_considered": 0,
            "rows_selected": 0,
            "rows_transferred": 0,
            "rejection_counts": {},
            "skipped": {"empty_source_split": 1},
            "class_counts": {},
            "status": "skipped",
            "reason": skip_reason,
        }
        summary["verification"] = verify_export(empty_manifest, destination_root)
        summary["threshold_source"] = threshold_source
        summary["manifest_path"] = manifest_path.as_posix()
        summary["rejected_path"] = rejected_path.as_posix()
        write_json(summary, summary_path)
        print("Status:            skipped")
        print("Files transferred: 0")
        print(f"Manifest:          {manifest_path}")
        print(f"Summary:           {summary_path}")
        return

    if args.dry_run:
        kept, rejected = select_confident_known_rows(
            frame, config.classes, threshold
        )
        print(f"Rows considered: {len(frame)}")
        print(f"Rows selected:   {len(kept)}")
        print("Rejections by reason:")
        for reason, count in rejected["selection_reason"].value_counts().sort_index().items():
            print(f"  {reason}: {count}")
        print("Selected per class:")
        for class_name, count in kept["source_label"].value_counts().sort_index().items():
            print(f"  {class_name}: {count}")
        print(f"Destination would be: {destination_root.as_posix()}")
        return

    manifest, rejected, summary = export_confident_subset(
        frame,
        config.classes,
        threshold,
        source_root,
        destination_root,
        mode=args.mode,
    )

    # Persist the intended plan and its rationale before any further work so a
    # moved file always has a recorded origin.
    atomic_write_csv(manifest, manifest_path)
    atomic_write_csv(rejected, rejected_path)

    verification = verify_export(manifest, destination_root)
    summary["verification"] = verification
    summary["threshold_source"] = threshold_source
    summary["manifest_path"] = manifest_path.as_posix()
    summary["rejected_path"] = rejected_path.as_posix()
    write_json(summary, summary_path)

    print(f"Mode:              {summary['mode']}")
    print(f"Rows considered:   {summary['rows_considered']}")
    print(f"Rows selected:     {summary['rows_selected']}")
    print(f"Files transferred: {summary['rows_transferred']}")
    for class_name, count in summary["class_counts"].items():
        print(f"  {class_name}: {count}")
    print(f"Rejections:        {summary['rejection_counts']}")
    if summary["skipped"]:
        print(f"Skipped:           {summary['skipped']}")
    print(f"Verified:          {verification['verified']}")
    print(f"Manifest:          {manifest_path}")
    print(f"Summary:           {summary_path}")


if __name__ == "__main__":
    main()