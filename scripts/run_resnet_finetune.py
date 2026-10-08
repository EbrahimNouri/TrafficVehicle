"""Train only the ResNet18 fine-tuning experiment and report its best checkpoint.

`python main.py` runs the whole 13-experiment protocol before it can name a
winner. This runner isolates the validation-winning `resnet18_fine_tuning` arm
so it can be trained, resumed, or simply re-reported on its own:

* it rebuilds the exact split, transforms, TrainOptions and metadata the
  pipeline used, so a compatible completed checkpoint is reused instead of
  retrained, and an existing checkpoint is never silently overwritten;
* validation only - `dataset/test` is never read;
* it refreshes the experiment's entry in
  `artifacts/results/experiment_results.json` so the recorded SHA-256 and
  metrics stay consistent for `scripts/promote_best_checkpoint.py`;
* if the checkpoint actually changes, the production bundle and the promoted
  copies are invalidated, mirroring the pipeline's start-of-run rule.

Run it from the project root (paths resolve against it regardless).

Examples:
    python scripts/run_resnet_finetune.py
    python scripts/run_resnet_finetune.py --epochs 60 --fresh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

from traffic_classifier.config import ProjectConfig, seed_everything  # noqa: E402
from traffic_classifier.data import (  # noqa: E402
    build_transforms,
    load_image_folder,
    make_loader,
    stratified_validation_indices,
)
from traffic_classifier.engine import (  # noqa: E402
    TrainOptions,
    _compatible_checkpoint,
    train_model,
)
from traffic_classifier.locking import ProjectRunLock, RunLockError  # noqa: E402
from traffic_classifier.models import (  # noqa: E402
    build_model,
    configure_resnet_stage,
    parameter_report,
)
from traffic_classifier.pipeline import (  # noqa: E402
    _confidence_summary,
    _strip_outputs,
    experiment_specs,
)
from traffic_classifier.promotion import invalidate_promotion, sha256_file  # noqa: E402
from traffic_classifier.utils import (  # noqa: E402
    ensure_directories,
    load_checkpoint,
    read_json,
    write_json,
)

EXPERIMENT = "resnet18_fine_tuning"


def _scaled_stage_epochs(
    spec_epochs: dict[str, int] | None, epochs: int
) -> dict[str, int]:
    """Mirror the pipeline's stage rescaling for a non-default epoch budget."""

    if spec_epochs is None:
        return {"head": epochs}
    stages = dict(spec_epochs)
    if epochs != 15:
        head = max(1, round(epochs * next(iter(stages.values())) / 15))
        stages = {"head": head, "layer4": epochs - head}
    return stages


def _options_diff(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Top-level TrainOptions keys whose values differ from the stored run."""

    stored = dict(stored)
    # `resume` controls startup behavior, not the trained configuration.
    stored["resume"] = current.get("resume")
    return sorted(
        key for key in set(stored) | set(current) if stored.get(key) != current.get(key)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train, resume, or re-report the ResNet18 fine-tuning experiment "
            "and print its best-validation checkpoint. Validation only; the "
            "frozen test split is never read."
        )
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the epoch budget (stage schedule rescales like the pipeline)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete this experiment's checkpoints and retrain from scratch",
    )
    args = parser.parse_args()

    config = ProjectConfig.from_json(ROOT / args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    seed_everything(config.seed)
    device = config.resolve_device()

    spec = next(
        (item for item in experiment_specs() if item.name == EXPERIMENT), None
    )
    if spec is None:
        raise SystemExit(f"experiment {EXPERIMENT!r} is not defined by the pipeline")
    stage_epochs = _scaled_stage_epochs(spec.stage_epochs, config.epochs)

    ensure_directories(
        Path(config.artifacts_dir) / "checkpoints",
        Path(config.artifacts_dir) / "results",
    )

    print("=== ResNet18 fine-tuning runner ===")
    print(f"config   : {args.config}")
    print(f"device   : {device}")
    print(f"epochs   : {config.epochs} (stages: {stage_epochs})")

    manifest_path = Path(config.artifacts_dir) / "run_manifest.json"
    run_fingerprint = (
        read_json(manifest_path).get("run_fingerprint")
        if manifest_path.is_file()
        else None
    )
    if run_fingerprint is None:
        print(
            "warning  : no run_manifest.json yet; the checkpoint will carry no "
            "run fingerprint (run `python main.py` to create one)"
        )
    else:
        print(f"run      : {run_fingerprint[:12]}...")

    # --- data: identical split, transforms and loaders to the pipeline arm ---
    train_transform, train_spec = build_transforms(
        config, augmentation=True, kind="resnet"
    )
    eval_transform, eval_spec = build_transforms(
        config, augmentation=False, kind="resnet"
    )
    train_dataset = load_image_folder(
        config.data_dir, "train", train_transform, config.classes
    )
    validation_dataset = load_image_folder(
        config.data_dir, "train", eval_transform, config.classes
    )
    train_indices, validation_indices, _ = stratified_validation_indices(
        train_dataset, config.validation_fraction, config.seed
    )
    train_loader = make_loader(
        train_dataset,
        train_indices,
        training=True,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        validation_indices,
        training=False,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    print(
        f"data     : {len(train_indices)} train / "
        f"{len(validation_indices)} validation (seed {config.seed})"
    )

    # --- model and options, byte-compatible with the pipeline's arm ---------
    seed_everything(config.seed)
    model = build_model(
        spec.architecture,
        len(config.classes),
        dropout=spec.dropout,
        pooling=spec.pooling,  # type: ignore[arg-type]
        pretrained=config.pretrained and spec.architecture == "resnet18",
        download_weights=config.download_pretrained,
    )
    configure_resnet_stage(model, "head")
    parameters = parameter_report(model)
    print(
        f"model    : {parameters['trainable_parameters']} trainable / "
        f"{parameters['total_parameters']} parameters (head stage)"
    )

    checkpoint_path = (
        Path(config.artifacts_dir) / "checkpoints" / f"{spec.name}.pt"
    )
    metadata: dict[str, Any] = {
        "run_fingerprint": run_fingerprint,
        "display_name": spec.display_name,
        "description": spec.description,
        "category": spec.category,
        "dropout": spec.dropout,
        "pooling": spec.pooling,
        "augmentation": spec.augmentation,
        "imbalanced": spec.imbalanced,
        "balanced_batches": spec.balanced,
        "class_mapping": {
            name: index for index, name in enumerate(config.classes)
        },
        "transform": train_spec.to_dict(),
        "evaluation_transform": eval_spec.to_dict(),
        "dataset": {
            "train_indices": train_indices,
            "imbalanced_train_indices": train_indices,
            "validation_indices": validation_indices,
            "frozen_test_used": False,
        },
        "parameter_report_head_stage": parameters,
        "balanced_sampler": None,
    }
    options = TrainOptions(
        experiment_name=spec.name,
        architecture=spec.architecture,
        loss=spec.loss,
        epochs=config.epochs,
        seed=config.seed,
        learning_rate=config.transfer_head_lr,
        head_learning_rate=config.transfer_head_lr,
        backbone_learning_rate=config.transfer_backbone_lr,
        weight_decay=spec.weight_decay,
        scheduler=spec.scheduler,
        scheduler_step_size=max(1, config.epochs // 3),
        stage_epochs=stage_epochs,
        freeze_batchnorm=spec.architecture == "resnet18",
        resume=not args.fresh,
        checkpoint_path=str(checkpoint_path),
        metadata=metadata,
    )

    try:
        with ProjectRunLock(config.artifacts_dir):
            _run(
                config=config,
                spec_name=spec.name,
                display_name=spec.display_name,
                description=spec.description,
                category=spec.category,
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                options=options,
                checkpoint_path=checkpoint_path,
                fresh=args.fresh,
            )
    except RunLockError as error:
        raise SystemExit(
            f"a project run currently owns {config.artifacts_dir}; wait for it "
            f"to finish before training ({error})"
        ) from error


def _run(
    *,
    config: ProjectConfig,
    spec_name: str,
    display_name: str,
    description: str,
    category: str,
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    options: TrainOptions,
    checkpoint_path: Path,
    fresh: bool,
) -> None:
    payload = (
        load_checkpoint(checkpoint_path, map_location="cpu")
        if checkpoint_path.is_file()
        else None
    )
    stored_resume = payload.get("train_options", {}).get("resume") if payload else None
    if (
        options.resume
        and payload is not None
        and not _compatible_checkpoint(payload, options, config.classes)
    ):
        changed = _options_diff(payload.get("train_options", {}), options.to_dict())
        raise SystemExit(
            f"{checkpoint_path.as_posix()} exists but was trained with a "
            f"different configuration (changed: {', '.join(changed) or 'unknown'}). "
            "Refusing to overwrite the existing best checkpoint; re-run with "
            "--fresh to discard it and retrain from scratch."
        )

    before = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None

    trained = train_model(
        model,
        train_loader,
        validation_loader,
        options,
        class_names=config.classes,
        device=config.resolve_device(),
    )

    after = sha256_file(checkpoint_path)
    if before is None:
        status = "trained from scratch"
    elif before == after:
        status = "existing completed checkpoint reused; no training was needed"
    else:
        status = "checkpoint updated (trained or resumed)"

    if before != after:
        # The checkpoint weights moved: nothing built on top of them is valid
        # any more, exactly as at the start of a pipeline run.
        Path(config.artifacts_dir, "checkpoints", "final_model.pt").unlink(
            missing_ok=True
        )
        Path(config.artifacts_dir, "production_ready.json").unlink(missing_ok=True)
        invalidate_promotion(config)

    outputs = trained["best_validation"]["outputs"]
    record_options = options.to_dict()
    if before == after and stored_resume is not None:
        # Keep the recorded `resume` flag as the original run wrote it; the
        # value only describes startup behavior and would otherwise churn the
        # results file on every re-report of a reused checkpoint.
        record_options["resume"] = stored_resume
    results_path = Path(config.artifacts_dir) / "results" / "experiment_results.json"
    results = read_json(results_path) if results_path.is_file() else {}
    results[spec_name] = {
        "name": spec_name,
        "display_name": display_name,
        "description": description,
        "category": category,
        "architecture": options.architecture,
        "loss": options.loss,
        "best_epoch": trained["best_epoch"],
        "validation_metrics": _strip_outputs(trained["best_validation"]),
        "validation_uncertainty": _confidence_summary(
            outputs["probabilities"], outputs["targets"]
        ),
        "history": trained["history"],
        "elapsed_seconds": trained["elapsed_seconds"],
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": after,
        "options": record_options,
        "metadata": options.metadata or {},
    }
    write_json(results, results_path)

    metrics = results[spec_name]["validation_metrics"]
    print("\n=== Best checkpoint ===")
    print(f"status     : {status}")
    print(f"checkpoint : {checkpoint_path.as_posix()}")
    print(f"sha256     : {after}")
    print(
        f"best epoch : {trained['best_epoch']}/{options.epochs} "
        f"(val macro-F1 {metrics['macro_f1']:.4f}, "
        f"val accuracy {metrics['accuracy']:.4f})"
    )
    print(f"record     : {results_path.as_posix()} (entry refreshed)")
    if before != after:
        print(
            "next       : production bundle and promoted copies were invalidated; "
            "run `python main.py` to re-freeze the test evaluation, then "
            "`python scripts/promote_best_checkpoint.py` to re-promote"
        )
    else:
        print(
            "next       : `python scripts/promote_best_checkpoint.py "
            "--verify-only` checks the promoted copy"
        )


if __name__ == "__main__":
    main()
