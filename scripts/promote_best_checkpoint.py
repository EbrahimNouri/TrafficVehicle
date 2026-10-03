"""Promote the best validation checkpoint(s) into `artifacts/best_checkpoints/`.

The pipeline already performs this step automatically before it evaluates the
frozen test set. This tool exists for the cases where you want to rebuild the
promoted directory without retraining, for example after copying an artifacts
directory to another machine:

* it never trains, never evaluates, and never reads the test split;
* the winner is chosen by validation macro-F1 only, matching the pipeline;
* every promoted file is verified against the SHA-256 recorded in the manifest.

Examples:
    Python scripts/promote_best_checkpoint.py
    python scripts/promote_best_checkpoint.py --verify-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_classifier.config import ProjectConfig  # noqa: E402
from traffic_classifier.locking import ProjectRunLock, RunLockError  # noqa: E402
from traffic_classifier.paths import load_paths  # noqa: E402
from traffic_classifier.promotion import (  # noqa: E402
    BEST_MODEL_NAME,
    best_checkpoints_dir,
    load_promoted_checkpoint,
    promote_best_checkpoints,
    read_manifest,
    sha256_file,
    verify_complete,
)
from traffic_classifier.utils import read_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy every experiment's best-validation checkpoint into "
            "artifacts/best_checkpoints/ and promote the validation winner to "
            "best_model.pt. Never touches the test split."
        )
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only re-verify existing promoted files against the manifest.",
    )
    args = parser.parse_args()

    paths = load_paths()
    config = ProjectConfig.from_json(ROOT / args.config)
    directory = best_checkpoints_dir(config)
    print(f"config        : {args.config}")
    print(f"target dir    : {paths.relative(directory)}")

    if args.verify_only:
        manifest = read_manifest(config)
        print(f"selected      : {manifest['selected_experiment']}")
        load_promoted_checkpoint(config, manifest=manifest)
        for name in sorted(manifest["experiments"]):
            load_promoted_checkpoint(config, name=f"{name}.pt", manifest=manifest)
        print(
            f"verified {len(manifest['experiments']) + 1} promoted checkpoints "
            "against the manifest hashes"
        )
        return

    results_path = Path(config.artifacts_dir) / "results" / "experiment_results.json"
    if not results_path.is_file():
        raise SystemExit(
            f"missing experiment record: {results_path}. Run `python main.py` first."
        )
    experiment_results = read_json(results_path)
    manifest_path = Path(config.artifacts_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"missing {manifest_path}; run `python main.py` before promoting."
        )
    run_fingerprint = read_json(manifest_path)["run_fingerprint"]

    # Fail closed when a pipeline run is still writing the record, or when the
    # record belongs to an older run. A partial record would crown the wrong
    # experiment, because the missing arms are the ones that might have won.
    verify_complete(
        experiment_results, run_fingerprint=run_fingerprint
    )
    try:
        with ProjectRunLock(config.artifacts_dir):
            pass
    except RunLockError as error:
        raise SystemExit(
            f"a project run currently owns {config.artifacts_dir}; wait for it to "
            f"finish before promoting ({error})"
        ) from error

    manifest = promote_best_checkpoints(
        config=config,
        experiment_results=experiment_results,
        selected_name=manifest_selected(experiment_results, run_fingerprint),
        run_fingerprint=run_fingerprint,
        paths_root=ROOT,
    )

    print(f"\nselection rule: {manifest['selection_rule']}")
    print(
        f"promoted      : {len(manifest['experiments'])} experiment checkpoints "
        f"+ {BEST_MODEL_NAME}"
    )
    header = f"{'val macro-F1':>12}  {'epoch':>5}  {'production':>9}  experiment"
    print(f"\n{header}\n{'-' * len(header)}")
    for name, entry in sorted(
        manifest["experiments"].items(),
        key=lambda item: -item[1]["validation_macro_f1"],
    ):
        marker = "  *" if entry["is_production"] else ""
        print(
            f"{entry['validation_macro_f1']:>12.4f}  {entry['best_epoch']:>5}  "
            f"{'yes' if entry['is_production'] else '':>9}  {name}{marker}"
        )
    print(
        f"\nproduction weights: {(directory / BEST_MODEL_NAME).name} "
        f"(sha256 {manifest['production_checkpoint_sha256']})"
    )
    print(f"manifest          : {paths.relative(directory / 'manifest.json')}")

    # Prove the promoted production weights load and match their recorded hash.
    load_promoted_checkpoint(config, manifest=manifest)
    actual = sha256_file(directory / BEST_MODEL_NAME)
    if actual != manifest["production_checkpoint_sha256"]:
        raise SystemExit("promoted production checkpoint failed hash verification")
    print("hash verification: ok")


def manifest_selected(experiment_results: dict, run_fingerprint: str) -> str:
    """Re-derive the validation winner using the pipeline's production rule."""

    candidates = [
        name
        for name, result in experiment_results.items()
        if result["loss"] == "cross_entropy" and result["category"] != "imbalance"
    ]
    if not candidates:
        raise SystemExit("no production candidates found in the experiment record")
    if run_fingerprint:
        matching = [
            name
            for name in candidates
            if experiment_results[name]["metadata"].get("run_fingerprint")
            == run_fingerprint
        ]
        if matching:
            candidates = matching
    return max(
        candidates,
        key=lambda name: experiment_results[name]["validation_metrics"]["macro_f1"],
    )


if __name__ == "__main__":
    main()