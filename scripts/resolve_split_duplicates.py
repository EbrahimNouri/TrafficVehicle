"""Resolve canonical-pixel duplicates that span the frozen train/test boundary.

The audit fails closed when a pixel-identical image exists in more than one
frozen split, because it cannot decide which copy is authoritative. This tool
makes that decision explicitly and reversibly: the copy in the frozen
evaluation split is always kept, and the copy in the training split is moved
into `dataset/quarantine/` rather than deleted. Every move is recorded in
`dataset/quarantine/resolved_split_duplicates.json` so the audit can report the
exclusion instead of silently claiming there were none.

Examples:
    Python scripts/resolve_split_duplicates.py --dry-run
    Python scripts/resolve_split_duplicates.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_classifier.config import ProjectConfig  # noqa: E402
from traffic_classifier.data import pixel_duplicate_groups, scan_split  # noqa: E402
from traffic_classifier.utils import write_json  # noqa: E402


# The split that defines the once-only evaluation sample. Its images are never
# moved: the frozen test must keep every image it was built with.
FROZEN_SPLIT = "test"
# The split that may lose a copy to the quarantine directory.
TRAINABLE_SPLIT = "train"
QUARANTINE_DIRNAME = "quarantine"
RESOLUTION_FILENAME = "resolved_split_duplicates.json"


def _resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def find_cross_split_duplicates(
    data_dir: Path, splits: tuple[str, ...], minimum_image_side: int
) -> list[dict[str, Any]]:
    """Group pixel-identical images that appear in more than one split."""

    records: list[dict[str, Any]] = []
    for split in splits:
        records.extend(scan_split(data_dir, split, minimum_image_side))
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(pixel_duplicate_groups(records), start=1):
        splits_present = {str(record["split"]) for record in group}
        if len(splits_present) < 2:
            continue
        groups.append(
            {
                "group": index,
                "pixel_sha256": group[0]["pixel_sha256"],
                "splits": sorted(splits_present),
                "label_conflict": len({str(record["label"]) for record in group}) > 1,
                "members": [
                    {
                        "split": str(record["split"]),
                        "label": str(record["label"]),
                        "path": Path(record["absolute_path"]).as_posix(),
                    }
                    for record in group
                ],
            }
        )
    return groups


def resolve(
    groups: list[dict[str, Any]],
    *,
    frozen_split: str = FROZEN_SPLIT,
    trainable_split: str = TRAINABLE_SPLIT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Plan one move per group: drop the trainable copy, keep the frozen copy.

    Returns `(moves, unresolved)`. A group is unresolved when it spans neither
    split in the expected way, so the caller can leave it to the fail-closed
    audit rather than guessing.
    """

    moves: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for group in groups:
        trainable = [m for m in group["members"] if m["split"] == trainable_split]
        frozen = [m for m in group["members"] if m["split"] == frozen_split]
        others = [
            m
            for m in group["members"]
            if m["split"] not in {trainable_split, frozen_split}
        ]
        if not trainable or not frozen or others or len(frozen) != 1:
            unresolved.append(group)
            continue
        source = Path(trainable[0]["path"])
        moves.append(
            {
                "group": group["group"],
                "pixel_sha256": group["pixel_sha256"],
                "label": trainable[0]["label"],
                "source_split": trainable_split,
                "kept_split": frozen_split,
                "decision": f"quarantine_{trainable_split}_copy_keep_{frozen_split}_copy",
                "source_path": source.as_posix(),
                "kept_duplicate_of": frozen[0]["path"],
                "label_conflict": group["label_conflict"],
            }
        )
    return moves, unresolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quarantine training-split copies of images that are pixel-identical "
            "to frozen test images"
        )
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.json"),
        help="Path to the project config JSON (default: <project>/configs/default.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the duplicate groups and planned moves without touching files",
    )
    args = parser.parse_args()

    config_path = _resolve(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}.")

    config = ProjectConfig.from_json(config_path)
    data_dir = Path(config.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    groups = find_cross_split_duplicates(
        data_dir, (TRAINABLE_SPLIT, FROZEN_SPLIT), config.minimum_image_side
    )
    print(f"Cross-split duplicate groups: {len(groups)}")
    if not groups:
        print(f"Nothing to resolve between {TRAINABLE_SPLIT} and {FROZEN_SPLIT}.")
        return

    moves, unresolved = resolve(groups)
    for move in moves:
        source = Path(move["source_path"])
        print(
            f"  group {move['group']:>3}  {TRAINABLE_SPLIT}/{move['label']}/"
            f"{source.name} -> {QUARANTINE_DIRNAME}/split_duplicates/"
            f"{move['label']}/{source.name}"
        )
    if unresolved:
        print(
            f"Unexplained groups left for the audit to fail on: {len(unresolved)}"
        )

    if args.dry_run:
        print(f"Planned moves: {len(moves)} (dry run, nothing written)")
        return

    quarantine_root = data_dir / QUARANTINE_DIRNAME / "split_duplicates"
    moved: list[dict[str, Any]] = []
    for move in moves:
        source = Path(move["source_path"])
        if not source.is_file():
            print(f"  skip (already gone): {source.as_posix()}")
            continue
        destination = quarantine_root / str(move["label"]) / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        moved.append({**move, "quarantine_path": destination.as_posix()})

    record = {
        "policy": (
            f"Keep the {FROZEN_SPLIT} copy, quarantine the {TRAINABLE_SPLIT} copy; "
            "no image is deleted."
        ),
        "frozen_split": FROZEN_SPLIT,
        "trainable_split": TRAINABLE_SPLIT,
        "groups_found": len(groups),
        "moved_count": len(moved),
        "unresolved_groups": len(unresolved),
        "moved": moved,
        "unresolved": unresolved,
    }
    record_path = data_dir / QUARANTINE_DIRNAME / RESOLUTION_FILENAME
    write_json(record, record_path)
    print(f"Moved:       {len(moved)}")
    print(f"Unresolved:  {len(unresolved)}")
    print(f"Record:      {record_path}")


if __name__ == "__main__":
    main()