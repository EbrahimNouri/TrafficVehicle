"""Promote the best validation checkpoint into a dedicated, verifiable directory.

Every experiment writes its best-validation checkpoint to
``artifacts/checkpoints/<experiment>.pt``. This module collects those files in
``artifacts/best_checkpoints/`` and records their provenance in a manifest, so
the weights used for testing can be identified by hash rather than by filename
convention.

The winner is chosen by validation macro-F1 only, which is the same rule the
pipeline uses for production selection. Nothing here looks at test scores:
promoting a checkpoint because it scored well on test would turn the frozen test
set into a selection tool.

``best_model.pt`` is a byte-identical copy of the winning experiment's
checkpoint, and :func:`load_promoted_checkpoint` refuses to load any promoted
file whose SHA-256 no longer matches its manifest entry.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from traffic_classifier.config import ProjectConfig
from traffic_classifier.utils import load_checkpoint, write_json

BEST_MODEL_NAME = "best_model.pt"
MANIFEST_NAME = "manifest.json"
SELECTION_RULE = (
    "highest validation macro-F1 among cross-entropy experiments, excluding the "
    "imbalance category; test scores are never used"
)


def sha256_file(path: str | Path) -> str:
    """Stream a SHA-256 over a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_checkpoints_dir(config: ProjectConfig) -> Path:
    # return Path(config.artifacts_dir) / "best_checkpoints"
    return Path("C:\\Users\\viroo\\PycharmProjects\\TrafficVehicle\\artifacts\\best_checkpoints")


def invalidate_promotion(config: ProjectConfig) -> None:
    """Drop the promoted production weights so no consumer can serve them.

    Called at the start of a run, next to the existing `final_model.pt`
    invalidation, so a superseded or half-written promotion is never loaded.
    """

    best_checkpoints_dir(config).mkdir(parents=True, exist_ok=True)
    (best_checkpoints_dir(config) / BEST_MODEL_NAME).unlink(missing_ok=True)
    (best_checkpoints_dir(config) / MANIFEST_NAME).unlink(missing_ok=True)


def verify_complete(
    experiment_results: dict[str, Any],
    *,
    expected: set[str] | None = None,
    run_fingerprint: str | None = None,
) -> None:
    """Refuse to promote a partial or stale experiment record.

    A pipeline run rewrites `experiment_results.json` incrementally, so a reader
    that catches it mid-run sees a valid-looking subset. Promoting from that
    subset would silently crown the wrong experiment, because the missing arms
    are exactly the ones that might have won.
    """

    if expected is None:
        from .pipeline import experiment_specs

        expected = {spec.name for spec in experiment_specs()}

    present = set(experiment_results)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if missing or unexpected:
        raise RuntimeError(
            "refusing to promote an incomplete experiment record: "
            f"missing={missing} unexpected={unexpected}"
        )

    if run_fingerprint:
        stale = sorted(
            name
            for name, result in experiment_results.items()
            if (result.get("metadata") or {}).get("run_fingerprint")
            != run_fingerprint
        )
        if stale:
            raise RuntimeError(
                "refusing to promote checkpoints from a different run: "
                f"{stale} do not carry run fingerprint {run_fingerprint}"
            )


def promote_best_checkpoints(
    *,
    config: ProjectConfig,
    experiment_results: dict[str, Any],
    selected_name: str,
    run_fingerprint: str,
    paths_root: Path | None = None,
    expected: set[str] | None = None,
) -> dict[str, Any]:
    """Copy every experiment checkpoint and the winner into the promoted directory.

    Returns the manifest that was written.
    """

    verify_complete(
        experiment_results, expected=expected, run_fingerprint=run_fingerprint
    )

    if selected_name not in experiment_results:
        raise KeyError(
            f"selected experiment {selected_name!r} is absent from the experiment "
            f"record ({sorted(experiment_results)})"
        )

    directory = best_checkpoints_dir(config)
    directory.mkdir(parents=True, exist_ok=True)

    entries: dict[str, Any] = {}
    wanted: set[str] = set()
    for name, result in sorted(experiment_results.items()):
        source = Path(result["checkpoint_path"])
        if not source.is_file():
            raise FileNotFoundError(f"missing checkpoint for {name}: {source}")
        digest = sha256_file(source)
        destination = directory / f"{name}.pt"
        if not destination.is_file() or sha256_file(destination) != digest:
            shutil.copy2(source, destination)
        if sha256_file(destination) != digest:
            raise RuntimeError(
                f"promoted copy of {name} does not match the source hash; the "
                "copy may have been interrupted"
            )
        wanted.add(destination.name)
        entries[name] = {
            "source": _relative(source, paths_root),
            "promoted": _relative(destination, paths_root),
            "sha256": digest,
            "architecture": result["architecture"],
            "category": result["category"],
            "loss": result["loss"],
            "best_epoch": result["best_epoch"],
            "validation_macro_f1": float(
                result["validation_metrics"]["macro_f1"]
            ),
            "is_production": name == selected_name,
        }

    # Remove promoted files for experiments that no longer exist, so the
    # directory never advertises weights from a superseded run.
    for stale in directory.glob("*.pt"):
        if stale.name not in wanted and stale.name != BEST_MODEL_NAME:
            stale.unlink()

    selected_source = Path(experiment_results[selected_name]["checkpoint_path"])
    selected_digest = entries[selected_name]["sha256"]
    best_model = directory / BEST_MODEL_NAME
    shutil.copy2(selected_source, best_model)
    if sha256_file(best_model) != selected_digest:
        raise RuntimeError(
            "best_model.pt does not match the selected experiment's checkpoint hash"
        )

    manifest = {
        "selection_rule": SELECTION_RULE,
        "selected_experiment": selected_name,
        "run_fingerprint": run_fingerprint,
        "production_checkpoint": BEST_MODEL_NAME,
        "production_checkpoint_sha256": selected_digest,
        "production_source": entries[selected_name]["source"],
        "production_validation_macro_f1": entries[selected_name][
            "validation_macro_f1"
        ],
        "experiments": entries,
    }
    write_json(manifest, directory / MANIFEST_NAME)
    return manifest


def read_manifest(config: ProjectConfig) -> dict[str, Any]:
    """Load the promotion manifest, failing loudly when it is absent."""

    from traffic_classifier.utils import read_json

    path = best_checkpoints_dir(config) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no promotion manifest at {path}; run the pipeline or "
            "scripts/promote_best_checkpoint.py first"
        )
    return read_json(path)


def load_promoted_checkpoint(
    config: ProjectConfig,
    *,
    name: str = BEST_MODEL_NAME,
    map_location: Any = "cpu",
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a promoted checkpoint after verifying it against the manifest hash."""

    manifest = manifest if manifest is not None else read_manifest(config)
    directory = best_checkpoints_dir(config)

    if name == BEST_MODEL_NAME:
        expected = manifest["production_checkpoint_sha256"]
    else:
        entry = manifest["experiments"].get(name.removesuffix(".pt"))
        if entry is None:
            raise KeyError(
                f"{name!r} is not promoted; known: {sorted(manifest['experiments'])}"
            )
        expected = entry["sha256"]

    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"promoted checkpoint is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"promoted checkpoint {path} has hash {actual}, but the manifest "
            f"records {expected}. The file was modified after promotion."
        )
    return load_checkpoint(path, map_location=map_location)


def _relative(path: Path, paths_root: Path | None) -> str:
    if paths_root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(paths_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()