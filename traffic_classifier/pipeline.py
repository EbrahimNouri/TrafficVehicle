"""End-to-end audit, controlled experiments, frozen test, and artifact generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn

from .audit import audit_dataset
from .config import ProjectConfig, seed_everything
from .data import (
    BalancedBatchSampler,
    build_transforms,
    cleaned_unclean_indices,
    load_image_folder,
    make_loader,
    paths_for_subset,
    simulated_imbalance_indices,
    split_record,
    stratified_validation_indices,
)
from .engine import TrainOptions, evaluate, predict, train_model
from .metrics import (
    classification_metrics,
    confusion_pair_ranking,
    expected_calibration_error,
    fit_temperature,
    multiclass_nll,
    probabilities_from_logits,
    risk_coverage_curve,
    select_review_threshold,
    worst_classes,
)
from .locking import ProjectRunLock
from .models import (
    build_model,
    parameter_report,
    pretrained_resnet18_identity,
)
from .plotting import (
    plot_confusion_matrices,
    plot_error_gallery,
    plot_per_class_comparison,
    plot_representative_images,
    plot_review_coverage,
    plot_size_distributions,
    plot_training_curves,
    plot_unclean_analysis,
)
from .reporting import write_all_reports
from .utils import (
    ensure_directories,
    load_checkpoint,
    markdown_table,
    read_json,
    save_checkpoint,
    write_json,
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    display_name: str
    description: str
    architecture: str = "cnn"
    dropout: float = 0.0
    pooling: str = "max"
    augmentation: bool = True
    loss: str = "cross_entropy"
    weight_decay: float = 0.0
    scheduler: str = "none"
    transform_kind: str = "cnn"
    imbalanced: bool = False
    balanced: bool = False
    stage_epochs: dict[str, int] | None = None
    category: str = "ablation"


def experiment_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec("cnn_baseline", "CNN baseline", "Four-block CNN; augmentation; no dropout; max pooling; fixed LR; AdamW wd=0"),
        ExperimentSpec("no_augmentation", "No augmentation", "One-factor change from baseline: deterministic resize only", augmentation=False),
        ExperimentSpec("dropout_0_3", "Dropout p=0.3", "One-factor change from baseline: dropout=0.3", dropout=0.3),
        ExperimentSpec("dropout_0_5", "Dropout p=0.5", "One-factor change from baseline: dropout=0.5", dropout=0.5),
        ExperimentSpec("avg_pool", "Average pooling", "One-factor change from baseline: average pooling in all four pooling blocks", pooling="avg"),
        ExperimentSpec("weight_decay_1e-4", "Weight decay 1e-4", "One-factor change from baseline: AdamW weight_decay=1e-4", weight_decay=0.0001),
        ExperimentSpec("step_lr", "StepLR", "One-factor change from baseline: StepLR; validation is not used to tune test results", scheduler="step"),
        ExperimentSpec("bce_loss", "BCEWithLogitsLoss", "One-factor loss comparison; one-hot float targets and sigmoid comparison scores", loss="bce"),
        ExperimentSpec("best_regularized", "Best regularized + scheduled", "Factors selected from validation among the controlled regularization/scheduler arms", category="main"),
        ExperimentSpec("imbalanced_standard", "Imbalanced + standard batches", "Baseline architecture/training on reproducible imbalanced subset with standard shuffled batches", imbalanced=True, category="imbalance"),
        ExperimentSpec("imbalanced_balanced", "Imbalanced + balanced batches", "Same subset/model/budget; custom equal-class-per-batch sampler", imbalanced=True, balanced=True, category="imbalance"),
        ExperimentSpec("resnet18_feature_extraction", "ResNet18 feature extraction", "ImageNet-pretrained frozen backbone; train replacement head", architecture="resnet18", transform_kind="resnet", stage_epochs={"head": 15}, category="transfer"),
        ExperimentSpec("resnet18_fine_tuning", "ResNet18 fine-tuning", "Train head for 5 epochs, then unfreeze layer4 + head for 10 epochs", architecture="resnet18", transform_kind="resnet", stage_epochs={"head": 5, "layer4": 10}, category="transfer"),
    ]


def _strip_outputs(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "outputs"}


def _calibration_summary(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    one_hot = np.eye(probabilities.shape[1], dtype=float)[targets]
    return {
        "nll": multiclass_nll(probabilities, targets),
        "ece_10_bins": expected_calibration_error(probabilities, targets, bins=10),
        "brier_multiclass": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "accuracy": float((probabilities.argmax(axis=1) == targets).mean()),
    }


def _confidence_summary(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Confidence diagnostics valid for CE softmax or educational BCE sigmoid."""

    return {
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "median_confidence": float(np.median(probabilities.max(axis=1))),
        "ece_argmax_10_bins": expected_calibration_error(probabilities, targets, bins=10),
        "accuracy": float((probabilities.argmax(axis=1) == targets).mean()),
        "probability_interpretation": (
            "multiclass_softmax" if np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4) else "independent_sigmoid_scores"
        ),
    }


def _selective_metrics(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, Any]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accepted = confidence >= threshold
    reviewed = ~accepted
    return {
        "threshold": float(threshold),
        "accepted_count": int(accepted.sum()),
        "review_count": int(reviewed.sum()),
        "automatic_coverage": float(accepted.mean()),
        "review_rate": float(reviewed.mean()),
        "automatic_accuracy": float((predictions[accepted] == targets[accepted]).mean()) if accepted.any() else None,
        "review_accuracy": float((predictions[reviewed] == targets[reviewed]).mean()) if reviewed.any() else None,
        "overall_correctly_flagged": int(
            ((predictions != targets) & reviewed).sum()
        ),
        "total_errors": int((predictions != targets).sum()),
    }


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_threads": torch.get_num_threads(),
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(frame: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, **kwargs)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_run_manifest(
    config: ProjectConfig,
    split: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    source_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    resolved_device = config.resolve_device()
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "resolved_device": str(resolved_device),
        "gpu_name": (
            torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else None
        ),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    pretrained_identity = pretrained_resnet18_identity(
        enabled=config.pretrained,
        download=config.download_pretrained,
    )
    dataset_records = [
        {
            "path": row.path,
            "label": row.label,
            "width": row.width,
            "height": row.height,
            "pixel_sha256": row.pixel_sha256,
        }
        for row in frame.sort_values("dataset_path").itertuples(index=False)
    ]
    manifest = {
        "schema_version": 2,
        "config": config.to_dict(),
        "runtime": runtime,
        "pretrained_resnet18": pretrained_identity,
        "class_mapping": {name: index for index, name in enumerate(config.classes)},
        "train_indices": [item["index"] for item in split["train"]],
        "validation_indices": [item["index"] for item in split["validation"]],
        "dataset_pixel_manifest_sha256": _stable_hash(dataset_records),
        "source_sha256": source_hashes,
    }
    manifest["run_fingerprint"] = _stable_hash(manifest)
    return manifest


def _run_project_impl(
    config_path: str | Path = "configs/default.json",
    *,
    force: bool = False,
    epochs_override: int | None = None,
    transfer_size_override: int | None = None,
) -> dict[str, Any]:
    """Run the complete project. Test evaluation is guarded as a single final step."""

    config = ProjectConfig.from_json(config_path)
    if epochs_override is not None:
        config.epochs = epochs_override
    if transfer_size_override is not None:
        config.transfer_image_size = transfer_size_override
    # Invalidate publication before a new run mutates artifacts. The default
    # prediction API must never serve weights from a superseded/incomplete run.
    Path(config.artifacts_dir, "checkpoints", "final_model.pt").unlink(
        missing_ok=True
    )
    Path(config.artifacts_dir, "production_ready.json").unlink(missing_ok=True)
    config.save(Path(config.artifacts_dir) / "resolved_config.json")
    config.save(Path(config.reports_dir) / "reproducibility_config.json")
    ensure_directories(
        config.artifacts_dir,
        config.reports_dir,
        Path(config.artifacts_dir) / "checkpoints",
        Path(config.artifacts_dir) / "results",
        Path(config.artifacts_dir) / "splits",
    )
    seed_everything(config.seed)
    device = config.resolve_device()
    write_json(_environment(), Path(config.artifacts_dir) / "environment.json")

    print("\n=== 1. Dataset audit ===")
    audit = audit_dataset(config)
    summary = audit["summary"]
    frame: pd.DataFrame = audit["frame"]
    plot_size_distributions(frame, config.reports_dir)
    representative = audit["representative_paths"]
    plot_representative_images(
        representative,
        [Path(path).parent.name for path in representative],
        config.reports_dir,
    )

    print("\n=== 2. Leakage-safe fixed validation split ===")
    split_train_transform, split_train_spec = build_transforms(
        config, augmentation=True, kind="cnn"
    )
    _, split_eval_spec = build_transforms(
        config, augmentation=False, kind="cnn"
    )
    train_for_split = load_image_folder(
        config.data_dir, "train", split_train_transform, config.classes
    )
    train_indices, validation_indices, by_class = stratified_validation_indices(
        train_for_split, config.validation_fraction, config.seed
    )
    split = split_record(
        train_for_split,
        train_indices,
        validation_indices,
        split_train_spec,
        config.seed,
        config.validation_fraction,
    )
    split["by_class"] = by_class
    split["transforms"] = {
        "train": split_train_spec.to_dict(),
        "evaluation": split_eval_spec.to_dict(),
    }
    split["frozen_test"] = {
        "source": "dataset/test",
        "count": summary["splits"]["test"]["images"],
        "excluded_before_freeze": 0,
        "used_for_selection": False,
    }
    write_json(split, Path(config.artifacts_dir) / "splits" / "train_validation.json")
    run_manifest = _build_run_manifest(config, split, frame)
    run_fingerprint = run_manifest["run_fingerprint"]
    write_json(run_manifest, Path(config.artifacts_dir) / "run_manifest.json")
    print(f"Run fingerprint: {run_fingerprint}")

    imbalance_indices, imbalance_details = simulated_imbalance_indices(
        train_indices,
        train_for_split.targets,
        train_for_split.class_to_idx,
        config.imbalanced_classes,
        config.imbalanced_retained_per_minority_class,
        config.seed + 100,
    )
    imbalance_details["class_names"] = {str(key): value for key, value in train_for_split.class_to_idx.items()}
    imbalance_details["retained_paths"] = [
        Path(train_for_split.samples[index][0]).as_posix()
        for index in imbalance_details["retained_indices"]
    ]
    write_json(imbalance_details, Path(config.artifacts_dir) / "splits" / "simulated_imbalance.json")

    print("\n=== 3. Controlled experiments (validation only) ===")
    results_dir = Path(config.artifacts_dir) / "results"
    experiments_path = results_dir / "experiment_results.json"
    stored: dict[str, Any] = {}
    if experiments_path.exists() and not force:
        try:
            cached_results = read_json(experiments_path)
        except (OSError, json.JSONDecodeError):
            print("Ignoring unreadable experiment index; checkpoints remain authoritative")
            cached_results = {}
        stored = {
            name: result
            for name, result in cached_results.items()
            if result.get("metadata", {}).get("run_fingerprint") == run_fingerprint
        }
    results: dict[str, Any] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    specs_by_name = {spec.name: spec for spec in experiment_specs()}

    transform_cache: dict[tuple[str, bool], tuple[Any, Any, Any, Any, Any, Any]] = {}

    def get_transform_bundle(kind: str, augmentation: bool):
        key = (kind, augmentation)
        if key not in transform_cache:
            train_transform, train_spec = build_transforms(
                config, augmentation=augmentation, kind=kind  # type: ignore[arg-type]
            )
            eval_transform, eval_spec = build_transforms(
                config, augmentation=False, kind=kind  # type: ignore[arg-type]
            )
            train_dataset = load_image_folder(
                config.data_dir, "train", train_transform, config.classes
            )
            validation_dataset = load_image_folder(
                config.data_dir, "train", eval_transform, config.classes
            )
            test_dataset = load_image_folder(
                config.data_dir, "test", eval_transform, config.classes
            )
            transform_cache[key] = (
                train_dataset,
                validation_dataset,
                test_dataset,
                train_spec,
                eval_spec,
                eval_transform,
            )
        return transform_cache[key]

    balanced_sampler = BalancedBatchSampler(
        targets=train_for_split.targets,
        indices=imbalance_indices,
        num_classes=len(config.classes),
        batch_size=config.batch_size,
        seed=config.seed,
    )
    standard_steps = len(balanced_sampler)

    for spec in experiment_specs():
        if spec.name == "best_regularized":
            def best_result(names: list[str]) -> str:
                return max(
                    names,
                    key=lambda name: results[name]["validation_metrics"]["macro_f1"],
                )

            augmentation_run = best_result(["cnn_baseline", "no_augmentation"])
            dropout_run = best_result(
                ["cnn_baseline", "dropout_0_3", "dropout_0_5"]
            )
            decay_run = best_result(["cnn_baseline", "weight_decay_1e-4"])
            scheduler_run = best_result(["cnn_baseline", "step_lr"])
            selected_factors = {
                "augmentation": augmentation_run == "cnn_baseline",
                "dropout": {"cnn_baseline": 0.0, "dropout_0_3": 0.3, "dropout_0_5": 0.5}[dropout_run],
                "weight_decay": 1e-4 if decay_run == "weight_decay_1e-4" else 0.0,
                "scheduler": "step" if scheduler_run == "step_lr" else "none",
            }
            spec = ExperimentSpec(
                name="best_regularized",
                display_name="Best regularized + scheduled",
                description=(
                    "Validation-selected explicit combination: "
                    + json.dumps(selected_factors, sort_keys=True)
                ),
                dropout=float(selected_factors["dropout"]),
                augmentation=bool(selected_factors["augmentation"]),
                weight_decay=float(selected_factors["weight_decay"]),
                scheduler=str(selected_factors["scheduler"]),
                category="main",
            )
            specs_by_name[spec.name] = spec
        if spec.stage_epochs is not None:
            spec_epochs = dict(spec.stage_epochs)
            # Keep a short override coherent with the two-stage fine-tuning design.
            if config.epochs != 15:
                head_epochs = max(1, round(config.epochs * next(iter(spec_epochs.values())) / 15))
                spec_epochs = {"head": head_epochs, "layer4": config.epochs - head_epochs}
            spec = ExperimentSpec(**{**spec.__dict__, "stage_epochs": spec_epochs})
            specs_by_name[spec.name] = spec
        checkpoint_path = Path(config.artifacts_dir) / "checkpoints" / f"{spec.name}.pt"
        if (
            spec.name in stored
            and checkpoint_path.exists()
            and stored[spec.name].get("metadata", {}).get("run_fingerprint")
            == run_fingerprint
            and stored[spec.name].get("checkpoint_sha256")
            == _sha256_file(checkpoint_path)
            and not force
        ):
            print(f"Reusing completed experiment: {spec.name}")
            result = stored[spec.name]
        else:
            print(f"\nRunning {spec.display_name} [{spec.name}]")
            (
                train_dataset,
                validation_dataset,
                _,
                train_transform_spec,
                eval_transform_spec,
                _,
            ) = get_transform_bundle(spec.transform_kind, spec.augmentation)
            use_indices = imbalance_indices if spec.imbalanced else train_indices
            sampler = None
            if spec.balanced:
                sampler = BalancedBatchSampler(
                    targets=train_dataset.targets,
                    indices=use_indices,
                    num_classes=len(config.classes),
                    batch_size=config.batch_size,
                    seed=config.seed,
                )
            train_loader = make_loader(
                train_dataset,
                use_indices,
                training=True,
                batch_size=config.batch_size,
                seed=config.seed,
                num_workers=config.num_workers,
                balanced_sampler=sampler,
                steps_per_epoch=standard_steps if spec.imbalanced and not spec.balanced else None,
            )
            validation_loader = make_loader(
                validation_dataset,
                validation_indices,
                training=False,
                batch_size=config.batch_size,
                seed=config.seed,
                num_workers=config.num_workers,
            )
            seed_everything(config.seed)
            model = build_model(
                spec.architecture,
                len(config.classes),
                dropout=spec.dropout,
                pooling=spec.pooling,  # type: ignore[arg-type]
                pretrained=config.pretrained and spec.architecture == "resnet18",
                download_weights=config.download_pretrained,
            )
            if spec.architecture == "resnet18":
                from .models import configure_resnet_stage

                configure_resnet_stage(model, "head")
            parameters = parameter_report(model)
            learning_rate = (
                config.transfer_head_lr
                if spec.architecture == "resnet18"
                else config.learning_rate
            )
            options = TrainOptions(
                experiment_name=spec.name,
                architecture=spec.architecture,
                loss=spec.loss,
                epochs=config.epochs,
                seed=config.seed,
                learning_rate=learning_rate,
                head_learning_rate=(
                    config.transfer_head_lr if spec.architecture == "resnet18" else None
                ),
                backbone_learning_rate=(
                    config.transfer_backbone_lr if spec.architecture == "resnet18" else None
                ),
                weight_decay=spec.weight_decay,
                scheduler=spec.scheduler,
                scheduler_step_size=max(1, config.epochs // 3),
                stage_epochs=spec.stage_epochs,
                freeze_batchnorm=spec.architecture == "resnet18",
                resume=not force,
                checkpoint_path=str(checkpoint_path),
                metadata={
                    "run_fingerprint": run_fingerprint,
                    "display_name": spec.display_name,
                    "description": spec.description,
                    "category": spec.category,
                    "dropout": spec.dropout,
                    "pooling": spec.pooling,
                    "augmentation": spec.augmentation,
                    "imbalanced": spec.imbalanced,
                    "balanced_batches": spec.balanced,
                    "class_mapping": {name: index for index, name in enumerate(config.classes)},
                    "transform": train_transform_spec.to_dict(),
                    "evaluation_transform": eval_transform_spec.to_dict(),
                    "dataset": {
                        "train_indices": train_indices,
                        "imbalanced_train_indices": use_indices,
                        "validation_indices": validation_indices,
                        "frozen_test_used": False,
                    },
                    "parameter_report_head_stage": parameters,
                    "balanced_sampler": {
                        "batch_size": sampler.batch_size,
                        "samples_per_class": sampler.samples_per_class,
                        "batches_per_epoch": len(sampler),
                        "total_samples_per_epoch": sampler.total_size,
                    }
                    if sampler is not None
                    else None,
                },
            )
            trained = train_model(
                model,
                train_loader,
                validation_loader,
                options,
                class_names=config.classes,
                device=device,
            )
            result = {
                "name": spec.name,
                "display_name": spec.display_name,
                "description": spec.description,
                "category": spec.category,
                "architecture": spec.architecture,
                "loss": spec.loss,
                "best_epoch": trained["best_epoch"],
                "validation_metrics": _strip_outputs(trained["best_validation"]),
                "validation_uncertainty": _confidence_summary(
                    trained["best_validation"]["outputs"]["probabilities"],
                    trained["best_validation"]["outputs"]["targets"],
                ),
                "history": trained["history"],
                "elapsed_seconds": trained["elapsed_seconds"],
                "checkpoint_path": checkpoint_path.as_posix(),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "options": options.to_dict(),
                "metadata": options.metadata,
            }
            stored[spec.name] = result
            write_json(stored, experiments_path)
        results[spec.name] = result
        histories[spec.name] = result["history"]

    experiment_frame = pd.DataFrame(
        [
            {
                "experiment": name,
                "display_name": result["display_name"],
                "category": result["category"],
                "architecture": result["architecture"],
                "loss": result["loss"],
                "best_epoch": result["best_epoch"],
                "validation_accuracy": result["validation_metrics"]["accuracy"],
                "validation_macro_precision": result["validation_metrics"]["macro_precision"],
                "validation_macro_recall": result["validation_metrics"]["macro_recall"],
                "validation_macro_f1": result["validation_metrics"]["macro_f1"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
            for name, result in results.items()
        ]
    ).sort_values("validation_macro_f1", ascending=False)
    _atomic_write_csv(
        experiment_frame, results_dir / "experiment_summary.csv"
    )
    write_json(stored, experiments_path)

    # Transfer details and sampler comparison.
    for name in ("resnet18_feature_extraction", "resnet18_fine_tuning"):
        result = results[name]
        write_json(
            {
                "experiment": name,
                "head_stage_parameter_report": result["metadata"]["parameter_report_head_stage"],
                "epochs_and_parameter_counts": [
                    {
                        "epoch": row["epoch"],
                        "stage": row["stage"],
                        "learning_rates": row["learning_rates"],
                        "trainable_parameters": row["trainable_parameters"],
                    }
                    for row in result["history"]
                ],
            },
            results_dir / f"{name}_training_details.json",
        )
    write_json(
        {
            "retained_counts": imbalance_details["class_counts"],
            "class_names": imbalance_details["class_names"],
            "retained_indices": imbalance_details["retained_indices"],
            "batch_size": balanced_sampler.batch_size,
            "samples_per_class": balanced_sampler.samples_per_class,
            "standard_and_balanced_steps_per_epoch": standard_steps,
            "standard": _strip_outputs(results["imbalanced_standard"]["validation_metrics"]),
            "balanced": _strip_outputs(results["imbalanced_balanced"]["validation_metrics"]),
        },
        results_dir / "balanced_batch_comparison.json",
    )

    print("\n=== 4. Select final model from validation only ===")
    production_candidates = [
        name
        for name in results
        if results[name]["loss"] == "cross_entropy"
        and results[name]["category"] != "imbalance"
    ]
    selected_name = max(
        production_candidates,
        key=lambda name: results[name]["validation_metrics"]["macro_f1"],
    )
    selected_spec = specs_by_name[selected_name]
    print(
        f"Selected {selected_name} with validation macro-F1 "
        f"{results[selected_name]['validation_metrics']['macro_f1']:.4f}"
    )

    selected_checkpoint_path = Path(results[selected_name]["checkpoint_path"])
    selected_checkpoint_sha256 = _sha256_file(selected_checkpoint_path)
    selected_checkpoint = load_checkpoint(
        selected_checkpoint_path, map_location=device
    )
    selected_model = build_model(
        selected_spec.architecture,
        len(config.classes),
        dropout=selected_spec.dropout,
        pooling=selected_spec.pooling,  # type: ignore[arg-type]
        pretrained=False,
    )
    selected_model.load_state_dict(selected_checkpoint["model_state"])
    selected_model.to(device)

    selected_transform, selected_transform_spec = build_transforms(
        config,
        augmentation=False,
        kind=selected_spec.transform_kind,  # type: ignore[arg-type]
    )
    selected_validation_dataset = load_image_folder(
        config.data_dir, "train", selected_transform, config.classes
    )
    selected_test_dataset = load_image_folder(
        config.data_dir, "test", selected_transform, config.classes
    )
    selected_validation_loader = make_loader(
        selected_validation_dataset,
        validation_indices,
        training=False,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    validation_outputs = evaluate(
        selected_model,
        selected_validation_loader,
        criterion,
        device,
        loss_name="cross_entropy",
        num_classes=len(config.classes),
        class_names=config.classes,
        probability_kind="softmax",
    )["outputs"]
    temperature = fit_temperature(
        validation_outputs["logits"], validation_outputs["targets"]
    )
    calibrated_validation_probabilities = probabilities_from_logits(
        validation_outputs["logits"], "softmax", temperature
    )
    review = select_review_threshold(
        calibrated_validation_probabilities,
        validation_outputs["targets"],
        target_coverage=config.target_review_coverage,
    )
    coverage_curve = risk_coverage_curve(
        calibrated_validation_probabilities, validation_outputs["targets"]
    )
    write_json(
        {
            "temperature": temperature,
            "uncalibrated": _calibration_summary(
                validation_outputs["probabilities"], validation_outputs["targets"]
            ),
            "calibrated": _calibration_summary(
                calibrated_validation_probabilities, validation_outputs["targets"]
            ),
            "review_threshold": review,
            "risk_coverage_curve": coverage_curve,
        },
        results_dir / "validation_uncertainty.json",
    )
    plot_review_coverage(coverage_curve, config.reports_dir)

    print("\n=== 5. One frozen-test evaluation ===")
    frozen_marker = results_dir / "frozen_test_evaluation.json"
    final_result: dict[str, Any]
    cached_test_paths: list[str] | None = None
    if frozen_marker.exists() and not force:
        cached_final = read_json(frozen_marker)
        protocol = cached_final.get("evaluation_protocol", {})
        test_outputs_path = results_dir / "test_outputs.npz"
        test_errors_path = results_dir / "test_errors.csv"
        if not test_outputs_path.exists() or not test_errors_path.exists():
            raise RuntimeError(
                "Frozen-test marker exists but its hashed evidence archive is missing: "
                f"{test_outputs_path} / {test_errors_path}"
            )
        expected_protocol = {
            "run_fingerprint": run_fingerprint,
            "selected_checkpoint_sha256": selected_checkpoint_sha256,
            "test_outputs_sha256": _sha256_file(test_outputs_path),
            "test_errors_sha256": _sha256_file(test_errors_path),
        }
        mismatches = [
            key
            for key, expected in expected_protocol.items()
            if protocol.get(key) != expected
        ]
        if (
            cached_final.get("selected_experiment") != selected_name
            or abs(float(cached_final.get("temperature", -1.0)) - temperature) > 1e-9
            or abs(
                float(cached_final.get("review_threshold", {}).get("threshold", -1.0))
                - float(review["threshold"])
            )
            > 1e-9
            or mismatches
        ):
            raise RuntimeError(
                "A frozen-test result already exists for a different model, protocol, "
                "or evidence hash. Refusing to score test again implicitly; use a new "
                "artifacts directory or pass --force only for an intentional new evaluation."
            )
        print("Reusing the verified single frozen-test evaluation")
        final_result = cached_final
        with np.load(test_outputs_path, allow_pickle=False) as test_outputs:
            test_targets = test_outputs["targets"].copy()
            test_predictions = test_outputs["predictions"].copy()
            test_probabilities = test_outputs["probabilities"].copy()
            cached_test_paths = [str(value) for value in test_outputs["paths"].tolist()]
    else:
        # This is the only test-set evaluation in the project workflow.
        selected_test_loader = make_loader(
            selected_test_dataset,
            range(len(selected_test_dataset)),
            training=False,
            batch_size=config.batch_size,
            seed=config.seed,
            num_workers=config.num_workers,
        )
        test_evaluation = evaluate(
            selected_model,
            selected_test_loader,
            criterion,
            device,
            loss_name="cross_entropy",
            num_classes=len(config.classes),
            class_names=config.classes,
            probability_kind="softmax",
        )
        test_targets = test_evaluation["outputs"]["targets"]
        test_predictions = test_evaluation["outputs"]["predictions"]
        test_probabilities = probabilities_from_logits(
            test_evaluation["outputs"]["logits"], "softmax", temperature
        )
        test_metrics = classification_metrics(
            test_targets, test_predictions, config.classes
        )
        test_uncertainty = _calibration_summary(test_probabilities, test_targets)
        test_selective = _selective_metrics(
            test_probabilities, test_targets, review["threshold"]
        )
        pairs = confusion_pair_ranking(
            test_metrics["confusion_matrix_row_normalized"], config.classes
        )
        test_paths = [
            path.as_posix() for path in paths_for_subset(selected_test_loader.dataset, [])
        ]
        cached_test_paths = test_paths
        error_rows = []
        for index in np.flatnonzero(test_targets != test_predictions):
            top_order = np.argsort(test_probabilities[index])[::-1][:3]
            error_rows.append(
                {
                    "path": test_paths[index],
                    "true_class": config.classes[int(test_targets[index])],
                    "predicted_class": config.classes[int(test_predictions[index])],
                    "confidence": float(test_probabilities[index].max()),
                    "top3": [
                        {
                            "class": config.classes[int(class_index)],
                            "probability": float(test_probabilities[index, class_index]),
                        }
                        for class_index in top_order
                    ],
                    "needs_review": bool(
                        test_probabilities[index].max() < review["threshold"]
                    ),
                }
            )
        _atomic_write_csv(
            pd.DataFrame(error_rows), results_dir / "test_errors.csv"
        )
        test_outputs_path = results_dir / "test_outputs.npz"
        _atomic_save_npz(
            test_outputs_path,
            targets=test_targets,
            logits=test_evaluation["outputs"]["logits"],
            probabilities=test_probabilities,
            predictions=test_predictions,
            paths=np.asarray(test_paths),
        )
        test_outputs_sha256 = _sha256_file(test_outputs_path)
        test_errors_sha256 = _sha256_file(results_dir / "test_errors.csv")
        final_result = {
            "evaluation_protocol": {
                "selected_using": "validation macro-F1 only",
                "selected_experiment": selected_name,
                "test_evaluations": 1,
                "frozen_before_model_selection": True,
                "temperature_selected_on": "validation",
                "review_threshold_selected_on": "validation",
                "run_fingerprint": run_fingerprint,
                "selected_checkpoint_sha256": selected_checkpoint_sha256,
                "test_outputs_sha256": test_outputs_sha256,
                "test_errors_sha256": test_errors_sha256,
            },
            "selected_experiment": selected_name,
            "selection_validation_metrics": results[selected_name]["validation_metrics"],
            "temperature": temperature,
            "review_threshold": review,
            "test_metrics": test_metrics,
            "test_uncertainty": test_uncertainty,
            "test_selective": test_selective,
            "mutual_confusion_pairs": pairs,
            "error_count": len(error_rows),
        }
        write_json(final_result, frozen_marker)

    if cached_test_paths is None:
        raise RuntimeError("Test paths are unavailable while rebuilding the error gallery")
    # Always regenerate the visualization from hash-verified archive evidence.
    plot_error_gallery(
        cached_test_paths,
        test_targets,
        test_predictions,
        test_probabilities,
        config.classes,
        config.reports_dir,
        count=12,
    )

    print("\n=== 6. Cleaned unclean / unseen-class analysis ===")
    unclean_transform, _ = build_transforms(
        config, augmentation=False, kind=selected_spec.transform_kind  # type: ignore[arg-type]
    )
    unclean_dataset = load_image_folder(
        config.data_dir, "unclean", unclean_transform
    )
    all_unclean_base = load_image_folder(
        config.data_dir, "unclean", unclean_transform
    )
    all_train_base = load_image_folder(
        config.data_dir, "train", unclean_transform, config.classes
    )
    all_test_base = load_image_folder(
        config.data_dir, "test", unclean_transform, config.classes
    )
    retained_unclean_indices, exclusion_records = cleaned_unclean_indices(
        all_train_base, all_test_base, all_unclean_base
    )
    unclean_loader = make_loader(
        unclean_dataset,
        retained_unclean_indices,
        training=False,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    # Inference-only: source labels can include the unseen ninth `neysan` class,
    # which must never be passed as an eight-class Cross-Entropy target.
    unclean_raw = predict(
        selected_model,
        unclean_loader,
        device,
        probability_kind="softmax",
        temperature=temperature,
    )
    unclean_probabilities = unclean_raw["probabilities"]
    neysan_index = unclean_dataset.class_to_idx["neysan"]
    known_mask = unclean_raw["targets"] != neysan_index
    known_map = {
        unclean_dataset.class_to_idx[class_name]: model_index
        for model_index, class_name in enumerate(config.classes)
    }
    known_targets = np.asarray(
        [known_map[int(value)] for value in unclean_raw["targets"][known_mask]],
        dtype=int,
    )
    known_predictions = unclean_raw["predictions"][known_mask]
    known_probabilities = unclean_probabilities[known_mask]
    known_metrics = classification_metrics(
        known_targets, known_predictions, config.classes
    )
    neysan_predictions = unclean_raw["predictions"][~known_mask]
    neysan_confidence = unclean_probabilities[~known_mask].max(axis=1)
    neysan_review = neysan_confidence < review["threshold"]
    unclean_paths = [
        path.as_posix() for path in paths_for_subset(unclean_loader.dataset, [])
    ]
    prediction_counts = {
        config.classes[index]: int((unclean_raw["predictions"] == index).sum())
        for index in range(len(config.classes))
    }
    unclean_result = {
        "cleaning": {
            "original_images": len(unclean_dataset),
            "retained_images": len(retained_unclean_indices),
            "excluded_duplicates": len(exclusion_records),
            "known_images": int(known_mask.sum()),
            "unseen_neysan_images": int((~known_mask).sum()),
        },
        "known_class_metrics": known_metrics,
        "known_class_uncertainty": _calibration_summary(known_probabilities, known_targets),
        "unseen_neysan": {
            "mean_confidence": float(neysan_confidence.mean()),
            "median_confidence": float(np.median(neysan_confidence)),
            "review_count": int(neysan_review.sum()),
            "review_rate": float(neysan_review.mean()),
            "predicted_class_counts": {
                config.classes[index]: int((neysan_predictions == index).sum())
                for index in range(len(config.classes))
            },
        },
        "all_cleaned_prediction_counts": prediction_counts,
        "interpretation": (
            "neysan is an unseen class, not a ninth training label. Confidence is not "
            "a calibrated OOD score, so high-confidence known-class predictions can still "
            "be wrong; low-confidence cases are routed to human review."
        ),
    }
    write_json(unclean_result, results_dir / "unclean_analysis.json")
    unclean_rows = []
    for index, path in enumerate(unclean_paths):
        true_label = unclean_dataset.classes[int(unclean_raw["targets"][index])]
        prediction = config.classes[int(unclean_raw["predictions"][index])]
        confidence = float(unclean_probabilities[index].max())
        unclean_rows.append(
            {
                "path": path,
                "source_label": true_label,
                "predicted_class": prediction,
                "confidence": confidence,
                "needs_review": confidence < review["threshold"],
                "is_unseen_neysan": true_label == "neysan",
            }
        )
    _atomic_write_csv(
        pd.DataFrame(unclean_rows), results_dir / "unclean_predictions.csv"
    )
    plot_unclean_analysis(
        unclean_paths,
        unclean_raw["targets"],
        unclean_raw["predictions"],
        unclean_probabilities,
        config.classes,
        config.reports_dir,
        unseen_index=neysan_index,
        unseen_name="neysan",
    )

    print("\n=== 7. Reports and production checkpoint ===")
    final_checkpoint = selected_checkpoint
    final_checkpoint["metadata"] = {
        **selected_checkpoint.get("metadata", {}),
        "final_selection": {
            "selected_by": "maximum validation macro-F1 among Cross-Entropy production candidates",
            "selected_experiment": selected_name,
            "validation_metrics": results[selected_name]["validation_metrics"],
            "temperature": temperature,
            "review_threshold": review,
            "final_test_metrics": final_result["test_metrics"],
        },
    }
    final_checkpoint["format_version"] = 3
    final_checkpoint["run_fingerprint"] = run_fingerprint
    final_checkpoint["selected_experiment"] = selected_name
    final_checkpoint["source_checkpoint_sha256"] = selected_checkpoint_sha256
    final_checkpoint["threshold"] = review["threshold"]
    final_checkpoint["temperature"] = temperature
    final_checkpoint["transform"] = selected_transform_spec.to_dict()
    final_checkpoint["class_mapping"] = {
        name: index for index, name in enumerate(config.classes)
    }
    final_checkpoint["seed"] = config.seed
    final_checkpoint["configuration"] = config.to_dict()
    final_checkpoint_path = Path(config.artifacts_dir) / "checkpoints" / "final_model.pt"
    save_checkpoint(final_checkpoint_path, final_checkpoint)

    main_names = [
        "cnn_baseline",
        "imbalanced_balanced",
        "bce_loss",
        "best_regularized",
        "resnet18_feature_extraction",
        "resnet18_fine_tuning",
    ]
    main_names = [name for name in main_names if name in results]
    plot_per_class_comparison(
        {name: results[name]["validation_metrics"] for name in main_names},
        config.classes,
        config.reports_dir,
    )
    plot_confusion_matrices(
        final_result["test_metrics"], config.classes, config.reports_dir
    )
    plot_training_curves(histories, config.reports_dir)
    write_all_reports(
        config,
        audit_summary=summary,
        split=split,
        imbalance=imbalance_details,
        results=results,
        experiment_frame=experiment_frame,
        final_result=final_result,
        unclean_result=unclean_result,
        exclusion_records=exclusion_records,
    )
    write_json(
        {
            "run_fingerprint": run_fingerprint,
            "selected_experiment": selected_name,
            "final_checkpoint": final_checkpoint_path.as_posix(),
            "final_checkpoint_sha256": _sha256_file(final_checkpoint_path),
            "source_checkpoint_sha256": selected_checkpoint_sha256,
            "test_outputs_sha256": final_result["evaluation_protocol"]["test_outputs_sha256"],
            "test_errors_sha256": final_result["evaluation_protocol"]["test_errors_sha256"],
            "published_after_reports": True,
        },
        Path(config.artifacts_dir) / "production_ready.json",
    )

    print("\nProject complete. See reports/final_report.md and reports/final_report.html")
    return {
        "audit": audit,
        "results": results,
        "selected_experiment": selected_name,
        "final_test": final_result,
        "unclean": unclean_result,
    }


def run_project(
    config_path: str | Path = "configs/default.json",
    *,
    force: bool = False,
    epochs_override: int | None = None,
    transfer_size_override: int | None = None,
) -> dict[str, Any]:
    """Serialize writers for one artifacts directory and run the full workflow."""

    config = ProjectConfig.from_json(config_path)
    with ProjectRunLock(config.artifacts_dir):
        return _run_project_impl(
            config_path,
            force=force,
            epochs_override=epochs_override,
            transfer_size_override=transfer_size_override,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the traffic-vehicle classification project")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--force", action="store_true", help="Retrain and reevaluate every experiment")
    parser.add_argument("--epochs", type=int, default=None, help="Override all run lengths (debug use)")
    parser.add_argument("--transfer-size", type=int, default=None, help="Override ResNet image size")
    args = parser.parse_args()
    run_project(
        args.config,
        force=args.force,
        epochs_override=args.epochs,
        transfer_size_override=args.transfer_size,
    )


if __name__ == "__main__":
    main()
