"""Complete PyTorch training and evaluation loops."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from .config import dataloader_generator, seed_everything
from .data import paths_for_subset
from .metrics import (
    classification_metrics,
    probabilities_from_logits,
)
from .models import configure_resnet_stage, set_batchnorm_eval
from .utils import load_checkpoint, save_checkpoint


@dataclass
class TrainOptions:
    experiment_name: str
    architecture: str
    loss: str = "cross_entropy"
    epochs: int = 15
    seed: int = 42
    learning_rate: float = 0.003
    head_learning_rate: float | None = None
    backbone_learning_rate: float | None = None
    weight_decay: float = 0.0
    scheduler: str = "none"
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.3
    stage_epochs: dict[str, int] | None = None
    freeze_batchnorm: bool = False
    resume: bool = True
    checkpoint_path: str = "artifacts/checkpoints/model.pt"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _criterion_for_loss(loss_name: str) -> nn.Module:
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unknown loss: {loss_name}")


def _one_hot_if_needed(loss_name: str, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if loss_name == "cross_entropy":
        return labels
    if loss_name == "bce":
        return F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)
    raise ValueError(f"Unknown loss: {loss_name}")


def _optimizer_groups(
    model: nn.Module, options: TrainOptions
) -> list[dict[str, Any]]:
    if options.architecture == "resnet18":
        groups = [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and name.startswith("fc.")
                ],
                "lr": options.head_learning_rate or options.learning_rate,
                "weight_decay": options.weight_decay,
                "name": "head",
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and not name.startswith("fc.")
                ],
                "lr": options.backbone_learning_rate or options.learning_rate,
                "weight_decay": options.weight_decay,
                "name": "pretrained_backbone",
            },
        ]
        return [group for group in groups if group["params"]]
    return [
        {
            "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
            "lr": options.learning_rate,
            "weight_decay": options.weight_decay,
            "name": "model",
        }
    ]


def _make_scheduler(
    optimizer: torch.optim.Optimizer, options: TrainOptions
) -> StepLR | ReduceLROnPlateau | None:
    if options.scheduler == "none":
        return None
    if options.scheduler == "step":
        return StepLR(
            optimizer,
            step_size=max(1, options.scheduler_step_size),
            gamma=options.scheduler_gamma,
        )
    if options.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=2, min_lr=1e-6
        )
    raise ValueError(f"Unknown scheduler: {options.scheduler}")


def _learning_rates(optimizer: torch.optim.Optimizer) -> list[dict[str, float | str]]:
    return [
        {
            "group": str(group.get("name", index)),
            "lr": float(group["lr"]),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_name: str,
    num_classes: int,
    freeze_batchnorm: bool = False,
) -> dict[str, float]:
    model.train()
    if freeze_batchnorm:
        set_batchnorm_eval(model)
    total_loss = 0.0
    total_examples = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        targets = _one_hot_if_needed(loss_name, labels, num_classes)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        predictions = logits.detach().argmax(dim=1).cpu().numpy()
        all_targets.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions)
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    return {
        "loss": total_loss / total_examples,
        "accuracy": float(accuracy_score(targets, predictions)),
    }


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    probability_kind: str = "softmax",
    temperature: float = 1.0,
    include_paths: bool = False,
) -> dict[str, Any]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        all_logits.append(logits.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    logits_array = np.concatenate(all_logits)
    targets_array = np.concatenate(all_targets)
    probabilities = probabilities_from_logits(
        logits_array, kind=probability_kind, temperature=temperature
    )
    predictions = logits_array.argmax(axis=1)
    result: dict[str, Any] = {
        "targets": targets_array,
        "logits": logits_array,
        "probabilities": probabilities,
        "predictions": predictions,
        "accuracy": float(accuracy_score(targets_array, predictions)),
    }
    if include_paths:
        result["paths"] = [path.as_posix() for path in paths_for_subset(loader.dataset, [])]
    return result


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    *,
    loss_name: str,
    num_classes: int,
    class_names: list[str],
    probability_kind: str = "softmax",
    temperature: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        targets = _one_hot_if_needed(loss_name, labels, num_classes)
        loss = criterion(logits, targets)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_logits.append(logits.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    logits_array = np.concatenate(all_logits)
    targets_array = np.concatenate(all_targets)
    predictions = logits_array.argmax(axis=1)
    probabilities = probabilities_from_logits(
        logits_array, kind=probability_kind, temperature=temperature
    )
    metrics = classification_metrics(targets_array, predictions, class_names)
    metrics["loss"] = total_loss / total_examples
    metrics["outputs"] = {
        "targets": targets_array,
        "logits": logits_array,
        "probabilities": probabilities,
        "predictions": predictions,
    }
    return metrics


def _legacy_train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    options: TrainOptions,
    *,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """Train, validate, and checkpoint by best validation macro-F1."""

    seed_everything(options.seed)
    model.to(device)
    num_classes = len(class_names)
    probability_kind = "sigmoid" if options.loss == "bce" else "softmax"
    criterion = _criterion_for_loss(options.loss)
    stage_epochs = options.stage_epochs or {"head": options.epochs}
    if sum(stage_epochs.values()) != options.epochs:
        raise ValueError("Sum of stage epochs must equal total epochs")

    history: list[dict[str, Any]] = []
    best_macro_f1 = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    global_epoch = 0
    started = time.perf_counter()

    for stage, epochs_for_stage in stage_epochs.items():
        if options.architecture == "resnet18":
            configure_resnet_stage(model, stage)  # type: ignore[arg-type]
        groups = _optimizer_groups(model, options)
        optimizer = AdamW(groups, lr=options.learning_rate, weight_decay=options.weight_decay)
        scheduler = _make_scheduler(optimizer, options)
        if hasattr(model, "fc"):
            set_trainable_description = {
                "head": int(
                    sum(
                        parameter.numel()
                        for parameter in model.fc.parameters()
                        if parameter.requires_grad
                    )
                ),
                "total_trainable": int(
                    sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                ),
            }
        else:
            set_trainable_description = {
                "all": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
            }

        for _ in range(epochs_for_stage):
            global_epoch += 1
            if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(global_epoch - 1)  # type: ignore[attr-defined]
            learning_rates = _learning_rates(optimizer)
            train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                loss_name=options.loss,
                num_classes=num_classes,
                freeze_batchnorm=options.freeze_batchnorm,
            )
            validation_metrics = evaluate(
                model,
                validation_loader,
                criterion,
                device,
                loss_name=options.loss,
                num_classes=num_classes,
                class_names=class_names,
                probability_kind=probability_kind,
            )
            row = {
                "epoch": global_epoch,
                "stage": stage,
                "learning_rates": learning_rates,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_precision": validation_metrics["macro_precision"],
                "validation_macro_recall": validation_metrics["macro_recall"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "train_validation_loss_gap": validation_metrics["loss"] - train_metrics["loss"],
                "trainable_parameters": set_trainable_description,
            }
            history.append(row)
            print(
                f"[{options.experiment_name}] epoch {global_epoch:02d}/{options.epochs} "
                f"stage={stage} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['validation_loss']:.4f} "
                f"val_acc={row['validation_accuracy']:.3f} "
                f"val_macro_f1={row['validation_macro_f1']:.3f} "
                f"lr={learning_rates}"
            )

            if validation_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = float(validation_metrics["macro_f1"])
                best_epoch = global_epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(validation_metrics["loss"])
                else:
                    scheduler.step()

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    best_validation = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        loss_name=options.loss,
        num_classes=num_classes,
        class_names=class_names,
        probability_kind=probability_kind,
    )
    elapsed = time.perf_counter() - started
    payload = {
        "format_version": 1,
        "model_state": best_state,
        "experiment_name": options.experiment_name,
        "class_names": class_names,
        "class_to_idx": {name: index for index, name in enumerate(class_names)},
        "seed": options.seed,
        "best_epoch": best_epoch,
        "best_validation_metrics": {
            key: value for key, value in best_validation.items() if key != "outputs"
        },
        "history": history,
        "train_options": options.to_dict(),
        "metadata": options.metadata or {},
        "elapsed_seconds": elapsed,
    }
    save_checkpoint(options.checkpoint_path, payload)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(options.checkpoint_path),
    }


def _rolling_checkpoint_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(f"{path.stem}.last{path.suffix}")


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _capture_rng_state(loader: DataLoader[Any]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    sampler_generator = getattr(loader.sampler, "generator", None)
    if sampler_generator is not None:
        state["dataloader_sampler_generator"] = sampler_generator.get_state()
    return state


def _restore_rng_state(loader: DataLoader[Any], state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    sampler_generator = getattr(loader.sampler, "generator", None)
    if sampler_generator is not None and state.get("dataloader_sampler_generator") is not None:
        sampler_generator.set_state(state["dataloader_sampler_generator"])


def _compatible_checkpoint(
    payload: dict[str, Any], options: TrainOptions, class_names: list[str]
) -> bool:
    stored_options = dict(payload.get("train_options", {}))
    # `resume` controls startup behavior, not the trained model configuration. This
    # lets a run started with --force resume normally after an interruption.
    stored_options["resume"] = options.resume
    return (
        stored_options == options.to_dict()
        and payload.get("class_names") == class_names
        and payload.get("metadata", {}).get("run_fingerprint")
        == (options.metadata or {}).get("run_fingerprint")
    )


def _best_checkpoint_payload(
    *,
    options: TrainOptions,
    class_names: list[str],
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_macro_f1: float,
    history: list[dict[str, Any]],
    elapsed_seconds: float,
    training_complete: bool,
) -> dict[str, Any]:
    return {
        "format_version": 4,
        "checkpoint_kind": "best",
        "training_complete": training_complete,
        "model_state": best_state,
        "experiment_name": options.experiment_name,
        "class_names": class_names,
        "class_to_idx": {name: index for index, name in enumerate(class_names)},
        "seed": options.seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "history": history,
        "train_options": options.to_dict(),
        "metadata": options.metadata or {},
        "elapsed_seconds": elapsed_seconds,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    options: TrainOptions,
    *,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """Train with per-epoch atomic checkpoints and exact automatic resume.

    ``<experiment>.last.pt`` is rewritten after every epoch and includes model,
    optimizer, scheduler, best weights, history, elapsed time, and RNG/DataLoader
    state. ``<experiment>.pt`` always contains the best validation model so far.
    An existing compatible completed checkpoint is loaded without training.
    """

    seed_everything(options.seed)
    model.to(device)
    num_classes = len(class_names)
    probability_kind = "sigmoid" if options.loss == "bce" else "softmax"
    criterion = _criterion_for_loss(options.loss)
    stage_epochs = options.stage_epochs or {"head": options.epochs}
    if sum(stage_epochs.values()) != options.epochs:
        raise ValueError("Sum of stage epochs must equal total epochs")
    if any(count < 0 for count in stage_epochs.values()):
        raise ValueError("Stage epoch counts cannot be negative")

    final_path = Path(options.checkpoint_path)
    rolling_path = _rolling_checkpoint_path(final_path)
    if not options.resume:
        for path in (final_path, rolling_path):
            path.unlink(missing_ok=True)
        print(f"[{options.experiment_name}] resume disabled; starting a fresh run")

    # A complete best checkpoint is sufficient even if the process stopped just
    # before experiment_results.json was written.
    if options.resume and final_path.exists():
        try:
            final_payload = load_checkpoint(final_path, map_location=device)
        except Exception as error:
            print(f"[{options.experiment_name}] ignoring unreadable best checkpoint: {error}")
        else:
            if (
                _compatible_checkpoint(final_payload, options, class_names)
                and final_payload.get("training_complete")
                and len(final_payload.get("history", [])) == options.epochs
            ):
                model.load_state_dict(final_payload["model_state"])
                best_validation = evaluate(
                    model,
                    validation_loader,
                    criterion,
                    device,
                    loss_name=options.loss,
                    num_classes=num_classes,
                    class_names=class_names,
                    probability_kind=probability_kind,
                )
                elapsed = float(final_payload.get("elapsed_seconds", 0.0))
                print(
                    f"[{options.experiment_name}] reusing completed checkpoint "
                    f"from epoch {final_payload['best_epoch']}"
                )
                return {
                    "model": model,
                    "history": final_payload["history"],
                    "best_epoch": int(final_payload["best_epoch"]),
                    "best_validation": best_validation,
                    "elapsed_seconds": elapsed,
                    "checkpoint_path": str(final_path),
                    "resumed": True,
                }

    schedule: list[tuple[int, str]] = []
    for stage, count in stage_epochs.items():
        schedule.extend((stage, index) for index in range(count))
    if len(schedule) != options.epochs:
        raise ValueError("Expanded training schedule does not match total epochs")

    history: list[dict[str, Any]] = []
    best_macro_f1 = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    completed_epochs = 0
    elapsed_offset = 0.0
    resume_payload: dict[str, Any] | None = None

    if options.resume and rolling_path.exists():
        try:
            candidate = load_checkpoint(rolling_path, map_location=device)
        except Exception as error:
            print(f"[{options.experiment_name}] ignoring unreadable rolling checkpoint: {error}")
        else:
            candidate_history = list(candidate.get("history", []))
            candidate_completed = int(
                candidate.get("completed_epochs", len(candidate_history))
            )
            if (
                _compatible_checkpoint(candidate, options, class_names)
                and 0 <= candidate_completed <= options.epochs
                and len(candidate_history) == candidate_completed
                and candidate.get("best_state") is not None
            ):
                resume_payload = candidate
                history = candidate_history
                completed_epochs = candidate_completed
                best_macro_f1 = float(candidate.get("best_validation_macro_f1", -float("inf")))
                best_epoch = int(candidate.get("best_epoch", 0))
                best_state = candidate.get("best_state")
                elapsed_offset = float(candidate.get("elapsed_seconds", 0.0))
                model.load_state_dict(candidate["model_state"])
                print(
                    f"[{options.experiment_name}] resuming from epoch "
                    f"{completed_epochs}/{options.epochs}"
                )
            else:
                print(
                    f"[{options.experiment_name}] rolling checkpoint is from a "
                    "different configuration; starting fresh"
                )

    started = time.perf_counter()
    current_stage: str | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: StepLR | ReduceLROnPlateau | None = None
    set_trainable_description: dict[str, int] = {}
    rng_restored = resume_payload is None

    for global_epoch, (stage, _) in enumerate(schedule, start=1):
        if global_epoch <= completed_epochs:
            continue
        if stage != current_stage:
            if options.architecture == "resnet18":
                configure_resnet_stage(model, stage)  # type: ignore[arg-type]
            optimizer = AdamW(
                _optimizer_groups(model, options),
                lr=options.learning_rate,
                weight_decay=options.weight_decay,
            )
            scheduler = _make_scheduler(optimizer, options)
            if (
                resume_payload is not None
                and resume_payload.get("current_stage") == stage
            ):
                optimizer.load_state_dict(resume_payload["optimizer_state"])
                if scheduler is not None and resume_payload.get("scheduler_state") is not None:
                    scheduler.load_state_dict(resume_payload["scheduler_state"])
            current_stage = stage

            if hasattr(model, "fc"):
                set_trainable_description = {
                    "head": int(
                        sum(
                            parameter.numel()
                            for parameter in model.fc.parameters()
                            if parameter.requires_grad
                        )
                    ),
                    "total_trainable": int(
                        sum(
                            parameter.numel()
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        )
                    ),
                }
            else:
                set_trainable_description = {
                    "all": int(
                        sum(
                            parameter.numel()
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        )
                    )
                }

        if not rng_restored:
            _restore_rng_state(train_loader, resume_payload["rng_state"])
            rng_restored = True
        assert optimizer is not None

        if hasattr(train_loader, "batch_sampler") and hasattr(
            train_loader.batch_sampler, "set_epoch"
        ):
            train_loader.batch_sampler.set_epoch(global_epoch - 1)  # type: ignore[attr-defined]
        learning_rates = _learning_rates(optimizer)
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            loss_name=options.loss,
            num_classes=num_classes,
            freeze_batchnorm=options.freeze_batchnorm,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            loss_name=options.loss,
            num_classes=num_classes,
            class_names=class_names,
            probability_kind=probability_kind,
        )
        row = {
            "epoch": global_epoch,
            "stage": stage,
            "learning_rates": learning_rates,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_precision": validation_metrics["macro_precision"],
            "validation_macro_recall": validation_metrics["macro_recall"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "train_validation_loss_gap": validation_metrics["loss"] - train_metrics["loss"],
            "trainable_parameters": set_trainable_description,
        }
        history.append(row)
        print(
            f"[{options.experiment_name}] epoch {global_epoch:02d}/{options.epochs} "
            f"stage={stage} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['validation_loss']:.4f} "
            f"val_acc={row['validation_accuracy']:.3f} "
            f"val_macro_f1={row['validation_macro_f1']:.3f} "
            f"lr={learning_rates}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(validation_metrics["macro_f1"])
            best_epoch = global_epoch
            best_state = _cpu_state_dict(model)
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(validation_metrics["loss"])
            else:
                scheduler.step()

        completed_epochs = global_epoch
        elapsed = elapsed_offset + (time.perf_counter() - started)
        rolling_payload = {
            "format_version": 4,
            "checkpoint_kind": "rolling",
            "training_complete": completed_epochs == options.epochs,
            "completed_epochs": completed_epochs,
            "current_stage": stage,
            "model_state": _cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "best_state": best_state,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
            "history": history,
            "rng_state": _capture_rng_state(train_loader),
            "elapsed_seconds": elapsed,
            "experiment_name": options.experiment_name,
            "class_names": class_names,
            "class_to_idx": {name: index for index, name in enumerate(class_names)},
            "seed": options.seed,
            "train_options": options.to_dict(),
            "metadata": options.metadata or {},
        }
        save_checkpoint(rolling_path, rolling_payload)
        if best_epoch == completed_epochs:
            save_checkpoint(
                final_path,
                _best_checkpoint_payload(
                    options=options,
                    class_names=class_names,
                    best_state=best_state,
                    best_epoch=best_epoch,
                    best_macro_f1=best_macro_f1,
                    history=history,
                    elapsed_seconds=elapsed,
                    training_complete=completed_epochs == options.epochs,
                ),
            )

    if best_state is None:
        raise RuntimeError("Training produced no best checkpoint")
    model.load_state_dict(best_state)
    best_validation = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        loss_name=options.loss,
        num_classes=num_classes,
        class_names=class_names,
        probability_kind=probability_kind,
    )
    elapsed = elapsed_offset + (time.perf_counter() - started)
    final_payload = _best_checkpoint_payload(
        options=options,
        class_names=class_names,
        best_state=best_state,
        best_epoch=best_epoch,
        best_macro_f1=best_macro_f1,
        history=history,
        elapsed_seconds=elapsed,
        training_complete=completed_epochs == options.epochs,
    )
    final_payload["best_validation_metrics"] = {
        key: value for key, value in best_validation.items() if key != "outputs"
    }
    save_checkpoint(final_path, final_payload)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": best_validation,
        "elapsed_seconds": elapsed,
        "checkpoint_path": str(final_path),
        "resumed": resume_payload is not None,
    }
