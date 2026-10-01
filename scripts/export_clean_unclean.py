"""Export the confident known-class subset of `unclean` into a new split.

Reads the prediction table produced by the pipeline and materializes only the
images whose class is certain: a known training label that the model agrees
with, at or above the calibrated review threshold. The unseen `neysan` class is
never promoted to a training label.

Examples:
    Python scripts/export_clean_unclean.py --dry-run
    Python scripts/export_clean_unclean.py --mode copy
    Python scripts/export_clean_unclean.py --mode move --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_classifier.config import ProjectConfig  # noqa: E402
from traffic_classifier.curation import (  # noqa: E402
    export_confident_subset,
    select_confident_known_rows,
    verify_export,
)
from traffic_classifier.utils import (  # noqa: E402
    atomic_write_csv,
    read_json,
    write_json,
)


def _review_threshold(
    uncertainty_path: Path, override: float | None
) -> tuple[float, str]:
    if override is not None:
        return float(override), "command line override"
    payload = read_json(uncertainty_path)
    threshold = float(payload["review_threshold"]["threshold"])
    return threshold, uncertainty_path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export confident known-class images from a scored split"
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--predictions", default="artifacts/results/unclean_predictions.csv"
    )
    parser.add_argument(
        "--uncertainty", default="artifacts/results/validation_uncertainty.json"
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
    args = parser.parse_args()

    if args.mode == "move" and not args.force and not args.dry_run:
        parser.error("--mode move requires --force because it empties the source split")

    config = ProjectConfig.from_json(args.config)
    source_root = Path(config.data_dir) / args.source_split
    destination_root = Path(config.data_dir) / args.destination_split
    results_dir = Path(config.artifacts_dir) / "results"
    predictions_path = Path(args.predictions)
    uncertainty_path = Path(args.uncertainty)

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction table not found: {predictions_path}. Run the pipeline first."
        )
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source split does not exist: {source_root}")

    frame = pd.read_csv(predictions_path)
    threshold, threshold_source = _review_threshold(uncertainty_path, args.threshold)
    print(f"Review threshold: {threshold:.6f} (from {threshold_source})")

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
    manifest_path = results_dir / "cleaned_unclean_manifest.csv"
    summary_path = results_dir / "cleaned_unclean_summary.json"
    rejected_path = results_dir / "cleaned_unclean_rejected.csv"
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