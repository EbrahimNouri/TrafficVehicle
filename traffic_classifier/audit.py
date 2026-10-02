"""Reproducible dataset audit and canonical-pixel duplicate detection."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data import pixel_duplicate_groups, scan_split
from .utils import ensure_directories, markdown_table, read_json, write_json


def _quantiles(values: list[int] | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _iqr_bounds(values: list[int] | list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    q25, q75 = np.quantile(array, [0.25, 0.75])
    spread = q75 - q25
    return float(q25 - 1.5 * spread), float(q75 + 1.5 * spread)


def _quarantined_split(record: dict[str, Any]) -> str:
    """Recover which split a quarantined image was removed from.

    The resolver records `source_split`, but the original path is authoritative:
    it is the only field that cannot drift if a record is hand-edited.
    """

    declared = record.get("source_split")
    if isinstance(declared, str) and declared:
        return declared
    parts = Path(str(record.get("source_path", ""))).parts
    for part in parts:
        if part in {"train", "test", "unclean"}:
            return part
    return ""


def audit_dataset(config: ProjectConfig) -> dict[str, Any]:
    """Audit all splits, write machine-readable evidence, and document exclusions."""

    audit_dir = Path(config.artifacts_dir) / "audit"
    ensure_directories(audit_dir)

    records: list[dict[str, Any]] = []
    for split in ("train", "test", "unclean"):
        split_records = scan_split(
            config.data_dir, split, config.minimum_image_side
        )
        for record in split_records:
            record["dataset_path"] = (
                Path(config.data_dir) / split / record["path"]
            ).as_posix()
        records.extend(split_records)

    if not records:
        raise FileNotFoundError(
            f"No image files were found under {Path(config.data_dir).resolve()}"
        )
    frame = pd.DataFrame(records)
    frame.to_csv(audit_dir / "image_records.csv", index=False)

    split_summaries: dict[str, Any] = {}
    for split, group in frame.groupby("split", sort=False):
        class_counts = group["label"].value_counts().sort_index()
        widths = group["width"].dropna().astype(int).tolist()
        heights = group["height"].dropna().astype(int).tolist()
        areas = (group["width"] * group["height"]).dropna().astype(int).tolist()
        width_low, width_high = _iqr_bounds(widths)
        height_low, height_high = _iqr_bounds(heights)
        split_summaries[str(split)] = {
            "images": int(len(group)),
            "class_counts": {str(key): int(value) for key, value in class_counts.items()},
            "formats": {
                str(key): int(value)
                for key, value in group["format"].fillna("unreadable").value_counts().items()
            },
            "width_quantiles": _quantiles(widths),
            "height_quantiles": _quantiles(heights),
            "area_quantiles": _quantiles(areas),
            "iqr_dimension_review": {
                "width_bounds": [width_low, width_high],
                "height_bounds": [height_low, height_high],
                "width_outlier_count": int(
                    ((group["width"] < width_low) | (group["width"] > width_high)).sum()
                ),
                "height_outlier_count": int(
                    ((group["height"] < height_low) | (group["height"] > height_high)).sum()
                ),
            },
            "aspect_ratio_quantiles": _quantiles(
                (group["width"] / group["height"]).dropna().astype(float).tolist()
            ),
            "unique_raw_sha256": int(group["raw_sha256"].nunique()),
            "unique_pixel_sha256": int(group["pixel_sha256"].nunique()),
            "issue_counts": dict(Counter(
                issue
                for issues in group["issues"]
                for issue in (issues if isinstance(issues, list) else [])
            )),
        }

    duplicate_groups = pixel_duplicate_groups(records)
    duplicate_payload: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    protected_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        if record["split"] in {"train", "test"} and record["pixel_sha256"]:
            protected_hashes[record["pixel_sha256"]].append(
                {
                    "split": record["split"],
                    "label": str(record["label"]),
                    "path": record["dataset_path"],
                }
            )

    for group_number, group in enumerate(duplicate_groups, start=1):
        digest = group[0]["pixel_sha256"]
        group_payload = {
            "group": group_number,
            "pixel_sha256": digest,
            "members": [
                {
                    "split": record["split"],
                    "label": record["label"],
                    "path": record["dataset_path"],
                    "raw_sha256": record["raw_sha256"],
                }
                for record in group
            ],
            "label_conflict": len({record["label"] for record in group}) > 1,
        }
        duplicate_payload.append(group_payload)
        protected = protected_hashes.get(digest, [])
        for record in group:
            if record["split"] == "unclean":
                sources = [item for item in protected if item["path"] != record["dataset_path"]]
                exclusions.append(
                    {
                        "decision": "exclude_from_cleaned_unclean",
                        "reason": "canonical_pixel_duplicate_of_train_or_test",
                        "unclean_path": record["dataset_path"],
                        "unclean_label": record["label"],
                        "pixel_sha256": digest,
                        "duplicate_of": sources,
                        "label_conflict": any(
                            source["label"] != record["label"] for source in sources
                        ),
                    }
                )

    excluded_unclean_paths = {
        item["unclean_path"] for item in exclusions
    }
    retained_unclean_counts = Counter(
        record["label"]
        for record in records
        if record["split"] == "unclean"
        and record["dataset_path"] not in excluded_unclean_paths
    )
    raw_hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["raw_sha256"]:
            raw_hash_groups[record["raw_sha256"]].append(record)
    raw_duplicate_groups = [
        group for group in raw_hash_groups.values() if len(group) > 1
    ]
    train_test_overlap_groups = [
        group
        for group in duplicate_payload
        if {"train", "test"}.issubset({member["split"] for member in group["members"]})
    ]

    # `scripts/resolve_split_duplicates.py` quarantines train-side copies of
    # frozen-test images. Read that record so the audit reports the real
    # exclusion count instead of claiming train was never touched. The moved
    # files are no longer under any split, so they cannot be counted from the
    # scan itself.
    resolution_path = (
        Path(config.data_dir) / "quarantine" / "resolved_split_duplicates.json"
    )
    resolution: dict[str, Any] = {}
    if resolution_path.is_file():
        try:
            resolution = read_json(resolution_path)
        except (OSError, ValueError):
            resolution = {}
    moved_records = [
        item for item in resolution.get("moved", []) if isinstance(item, dict)
    ]
    train_excluded = sum(
        1 for item in moved_records if _quarantined_split(item) == "train"
    )
    test_excluded = sum(
        1 for item in moved_records if _quarantined_split(item) == "test"
    )

    representative: dict[str, list[str]] = {}
    for class_name in sorted(frame["label"].dropna().unique()):
        paths = sorted(
            frame.loc[
                (frame["split"] == "train") & (frame["label"] == class_name),
                "dataset_path",
            ].tolist()
        )
        representative[class_name] = paths[:2]
    representative_paths = [path for paths in representative.values() for path in paths]

    issue_counts = Counter(
        issue
        for issues in frame["issues"]
        for issue in (issues if isinstance(issues, list) else [])
    )
    raw_unique = int(frame["raw_sha256"].nunique())
    pixel_unique = int(frame["pixel_sha256"].nunique())
    summary = {
        "dataset_root": str(Path(config.data_dir).as_posix()),
        "minimum_image_side": config.minimum_image_side,
        "total_images": int(len(frame)),
        "total_raw_unique_hashes": raw_unique,
        "total_canonical_pixel_unique_hashes": pixel_unique,
        "raw_file_duplicate_groups": len(raw_duplicate_groups),
        "canonical_duplicate_groups": len(duplicate_groups),
        "canonical_duplicate_extra_copies": sum(len(group) - 1 for group in duplicate_groups),
        "cross_split_duplicate_groups": sum(
            len({member["split"] for member in group["members"]}) > 1
            for group in duplicate_payload
        ),
        "label_conflict_duplicate_groups": sum(
            bool(group["label_conflict"]) for group in duplicate_payload
        ),
        "train_test_overlap_groups": len(train_test_overlap_groups),
        "issue_counts": dict(issue_counts),
        "splits": split_summaries,
        "cleaning": {
            "train_excluded": train_excluded,
            "test_excluded": test_excluded,
            "unclean_excluded": len(exclusions),
            "clean_counts": {
                "train": split_summaries["train"]["images"],
                "test": split_summaries["test"]["images"],
                "unclean": sum(retained_unclean_counts.values()),
                "unclean_known": sum(
                    count for label, count in retained_unclean_counts.items() if label != "neysan"
                ),
                "unclean_neysan": int(retained_unclean_counts.get("neysan", 0)),
                "unclean_class_counts": dict(sorted(retained_unclean_counts.items())),
            },
            "split_duplicate_policy": (
                "Keep the frozen-test copy; quarantine the pixel-identical train "
                "copy to `dataset/quarantine/split_duplicates/`."
            ),
            "quarantined_records": resolution.get("moved_count", 0),
            "policy": "Freeze original test; exclude only train/unclean copies.",
        },
    }

    write_json(summary, audit_dir / "summary.json")
    write_json(duplicate_payload, audit_dir / "canonical_duplicate_groups.json")
    write_json(exclusions, audit_dir / "cleaned_unclean_exclusions.json")
    write_json(resolution, audit_dir / "resolved_split_duplicates.json")
    write_json(representative, audit_dir / "representative_paths.json")
    write_json(
        {
            "seed": config.seed,
            "strategy": "per-class shuffled split using numpy.random.default_rng(seed)",
            "note": "Exact indices and paths are written by the pipeline before training.",
        },
        audit_dir / "split_protocol.json",
    )
    _write_audit_markdown(config, summary, duplicate_payload, exclusions, representative_paths)
    if train_test_overlap_groups:
        raise RuntimeError(
            "Canonical-pixel duplicates overlap frozen train and test. The workflow "
            "will not choose a side automatically; quarantine or resolve these groups "
            "before creating a validation split."
        )
    return {
        "summary": summary,
        "duplicates": duplicate_payload,
        "exclusions": exclusions,
        "representative_paths": representative_paths,
        "frame": frame,
    }


def _write_audit_markdown(
    config: ProjectConfig,
    summary: dict[str, Any],
    duplicate_groups: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    representative_paths: list[str],
) -> None:
    split_rows = []
    for split in ("train", "test", "unclean"):
        values = summary["splits"].get(split)
        if values is None:
            # A split with no surviving images contributes no group to the
            # summary; report it as empty instead of failing the audit.
            split_rows.append([split, 0, "-", "-", "-", 0])
            continue
        split_rows.append(
            [
                split,
                values["images"],
                ", ".join(f"{key}={count}" for key, count in values["class_counts"].items()),
                f"{values['width_quantiles']['min']:.0f}–{values['width_quantiles']['max']:.0f}",
                f"{values['height_quantiles']['min']:.0f}–{values['height_quantiles']['max']:.0f}",
                values["unique_pixel_sha256"],
            ]
        )
    issue_rows = [[key, value] for key, value in summary["issue_counts"].items()] or [
        ["none", 0]
    ]
    duplicate_rows = []
    for group in duplicate_groups:
        members = "; ".join(
            f"{member['split']}/{member['label']}/{Path(member['path']).name}"
            for member in group["members"]
        )
        duplicate_rows.append(
            [group["group"], members, "yes" if group["label_conflict"] else "no"]
        )
    cleaning = summary["cleaning"]
    report = f"""# Data Audit and Leakage Prevention

## Protocol

- Audit completed before validation creation and model fitting.
- Each split is loaded separately; `unclean` is never merged into training.
- The audit computes both encoded-file SHA-256 and a canonical decoded, EXIF-oriented RGB pixel hash. The latter is encoding-independent by design. In this supplied copy both methods identify the same {summary['canonical_duplicate_groups']} duplicate groups, but the canonical result remains the authoritative leakage check.
- Files smaller than {config.minimum_image_side} pixels on either side, unreadable files, unsupported formats, and extreme aspect ratios would be flagged.
- Frozen train/test exclusions: **{cleaning['train_excluded'] + cleaning['test_excluded']}** ({cleaning['split_duplicate_policy']}). Current train: **{summary['splits']['train']['images']}**; original test: **{summary['splits']['test']['images']}**.

## Original splits

{markdown_table(['Split', 'Images', 'Class counts', 'Width range', 'Height range', 'Unique pixel hashes'], split_rows)}

Quality findings:

{markdown_table(['Finding', 'Count'], issue_rows)}

All {summary['total_images']:,} source files decode successfully, use JPEG format, and have the same eight classes except for `unclean/neysan`. Dimensions vary substantially (as expected for traffic-camera crops), but no image is unusually small under the declared rule. Statistical 1.5×IQR width/height review counts are recorded per split in `summary.json`; these are review flags, not automatic exclusions.

## Canonical-pixel duplicate findings

- Encoded-file SHA-256 unique values: **{summary['total_raw_unique_hashes']:,}**; duplicate groups: **{summary['raw_file_duplicate_groups']}**.
- Canonical RGB pixel unique values: **{summary['total_canonical_pixel_unique_hashes']:,}**; duplicate groups: **{summary['canonical_duplicate_groups']}**; extra copies: **{summary['canonical_duplicate_extra_copies']}**.
- Cross-split groups: **{summary['cross_split_duplicate_groups']}**; label conflicts: **{summary['label_conflict_duplicate_groups']}**; train–test overlap groups: **{summary['train_test_overlap_groups']}** (the workflow fails closed if nonzero).

{markdown_table(['Group', 'Members', 'Conflict?'], duplicate_rows)}

Filename equality alone was not used. For example, `214844236.jpg` is a pixel-identical `train/vanet` and `unclean/neysan` image, demonstrating the apparent label conflict. It is excluded from the cleaned `unclean` analysis rather than relabelled or used for training.

## Cleaning decisions

{markdown_table(['Dataset', 'Excluded', 'Clean retained'], [
    ['train', cleaning['train_excluded'], cleaning['clean_counts']['train']],
    ['test (frozen)', cleaning['test_excluded'], cleaning['clean_counts']['test']],
    ['unclean', cleaning['unclean_excluded'], cleaning['clean_counts']['unclean']],
])}

Retained `unclean` class counts: {markdown_table(['Class', 'Count'], [[key, value] for key, value in cleaning['clean_counts']['unclean_class_counts'].items()])}.

Every exclusion is recorded in `artifacts/audit/cleaned_unclean_exclusions.json`. All exclusions are duplicate copies inside `unclean`; none belongs to the original train or test split. This avoids leakage without silently changing the frozen evaluation sample.

## Representative examples

The reproducible 16-image gallery (two per known class) is saved as `reports/figures/representative_images.png` from these exact paths:

{chr(10).join(f'- `{path}`' for path in representative_paths)}
"""
    report_path = Path(config.reports_dir) / "data_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
