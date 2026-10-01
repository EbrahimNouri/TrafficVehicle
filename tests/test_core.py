from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from torchvision import datasets, transforms

from traffic_classifier.config import ProjectConfig
from traffic_classifier.curation import (
    export_confident_subset,
    select_confident_known_rows,
    verify_export,
)
from traffic_classifier.data import (
    BalancedBatchSampler,
    build_transforms,
    canonical_pixel_hash,
    make_loader,
    paths_for_subset,
)
from traffic_classifier.engine import (
    TrainOptions,
    _compatible_checkpoint,
    _one_hot_if_needed,
    _rolling_checkpoint_path,
)
from traffic_classifier.locking import ProjectRunLock, RunLockError
from traffic_classifier.models import TrafficCNN, parameter_report
from traffic_classifier.pipeline import experiment_specs


def test_balanced_sampler_has_equal_classes_and_is_deterministic() -> None:
    targets = []
    for class_index in range(8):
        targets.extend([class_index] * (14 if class_index in {1, 3} else 40))
    sampler = BalancedBatchSampler(
        targets=targets,
        indices=list(range(len(targets))),
        num_classes=8,
        batch_size=32,
        seed=42,
    )
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(0)
    second = list(sampler)
    assert first == second
    assert len(first) == 10
    assert all(len(batch) == 32 for batch in first)
    for batch in first:
        assert np.bincount(np.asarray(targets)[batch], minlength=8).tolist() == [4] * 8


def test_balanced_sampler_requires_divisible_batch_size() -> None:
    try:
        BalancedBatchSampler([0, 1], [0, 1], 2, batch_size=3)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_canonical_pixel_hash_ignores_container_encoding(tmp_path: Path) -> None:
    pixels = np.full((16, 20, 3), [17, 93, 201], dtype=np.uint8)
    png = tmp_path / "sample.png"
    bmp = tmp_path / "sample.bmp"
    Image.fromarray(pixels).save(png)
    Image.fromarray(pixels).save(bmp)
    assert canonical_pixel_hash(png) == canonical_pixel_hash(bmp)


def test_bce_targets_are_one_hot_float() -> None:
    labels = torch.tensor([0, 3, 7])
    encoded = _one_hot_if_needed("bce", labels, num_classes=8)
    assert encoded.shape == (3, 8)
    assert encoded.dtype == torch.float32
    assert torch.equal(encoded.argmax(dim=1), labels)


def test_traffic_cnn_shape_and_parameter_report() -> None:
    model = TrafficCNN(num_classes=8, dropout=0.3, pooling="avg")
    output = model(torch.randn(2, 3, 96, 96))
    assert output.shape == (2, 8)
    report = parameter_report(model)
    assert report["trainable_parameters"] == report["total_parameters"]
    assert report["frozen_parameters"] == 0
    assert report["trainable_percent"] == 100.0


def test_make_loader_integrates_parent_indices_with_balanced_sampler(tmp_path: Path) -> None:
    pixels = np.full((12, 12, 3), 128, dtype=np.uint8)
    for class_index in range(8):
        class_dir = tmp_path / "train" / f"class_{class_index}"
        class_dir.mkdir(parents=True)
        Image.fromarray(pixels).save(class_dir / "sample.png")
    dataset = datasets.ImageFolder(tmp_path / "train", transform=transforms.ToTensor())
    sampler = BalancedBatchSampler(
        dataset.targets, list(range(8)), num_classes=8, batch_size=8, seed=7
    )
    loader = make_loader(
        dataset,
        list(range(8)),
        training=True,
        batch_size=8,
        seed=7,
        balanced_sampler=sampler,
    )
    _, labels = next(iter(loader))
    assert np.bincount(labels.numpy(), minlength=8).tolist() == [1] * 8


def test_evaluation_transform_is_deterministic_and_documented() -> None:
    config = ProjectConfig()
    transform, spec = build_transforms(config, augmentation=False, kind="cnn")
    image = Image.fromarray(np.full((20, 30, 3), 64, dtype=np.uint8))
    first = transform(image)
    second = transform(image)
    assert torch.equal(first, second)
    assert not any("Random" in operation for operation in spec.operations)
    assert any("Normalize" in operation for operation in spec.operations)


def test_controlled_experiment_factors_are_distinct() -> None:
    specs = {spec.name: spec for spec in experiment_specs()}
    assert specs["no_augmentation"].augmentation is False
    assert specs["dropout_0_3"].dropout == 0.3
    assert specs["dropout_0_5"].dropout == 0.5
    assert specs["avg_pool"].pooling == "avg"
    assert specs["weight_decay_1e-4"].weight_decay == 1e-4
    assert specs["step_lr"].scheduler == "step"
    assert specs["bce_loss"].loss == "bce"


def test_project_run_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    with ProjectRunLock(tmp_path):
        try:
            with ProjectRunLock(tmp_path):
                raise AssertionError("Concurrent writer unexpectedly acquired lock")
        except RunLockError:
            pass
    with ProjectRunLock(tmp_path):
        pass


def test_checkpoint_resume_compatibility_ignores_startup_flag(tmp_path: Path) -> None:
    classes = [f"class_{index}" for index in range(8)]
    metadata = {"run_fingerprint": "same-run"}
    interrupted = TrainOptions(
        experiment_name="demo",
        architecture="cnn",
        resume=False,
        checkpoint_path=str(tmp_path / "demo.pt"),
        metadata=metadata,
    )
    normal_restart = TrainOptions(
        experiment_name="demo",
        architecture="cnn",
        resume=True,
        checkpoint_path=str(tmp_path / "demo.pt"),
        metadata=metadata,
    )
    payload = {
        "train_options": interrupted.to_dict(),
        "class_names": classes,
        "metadata": metadata,
    }
    assert _compatible_checkpoint(payload, normal_restart, classes)
    assert _rolling_checkpoint_path(tmp_path / "demo.pt").name == "demo.last.pt"


def test_paths_respect_subset_positions(tmp_path: Path) -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    class_dir = tmp_path / "class"
    class_dir.mkdir()
    for index in range(3):
        Image.fromarray(pixels + index).save(class_dir / f"{index}.png")
    dataset = datasets.ImageFolder(tmp_path, transform=transforms.ToTensor())
    from torch.utils.data import Subset

    subset = Subset(dataset, [2, 0])
    assert [path.name for path in paths_for_subset(subset, [])] == ["2.png", "0.png"]
    assert [path.name for path in paths_for_subset(subset, [1])] == ["0.png"]


def _write_prediction_table(
    source_root: Path,
    rows: list[tuple[str, str, str, float, bool, bool]],
) -> pd.DataFrame:
    """Create fake split images plus the prediction table that scores them."""

    records = []
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    for index, (label, prediction, confidence, review, unseen) in enumerate(rows):
        class_dir = source_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        name = f"{label}_{index}.png"
        Image.fromarray(pixels + index).save(class_dir / name)
        records.append(
            {
                "path": (source_root / label / name).as_posix(),
                "source_label": label,
                "predicted_class": prediction,
                "confidence": confidence,
                "needs_review": review,
                "is_unseen_neysan": unseen,
            }
        )
    return pd.DataFrame(records)


def test_selection_keeps_only_agreed_confident_known_rows(tmp_path: Path) -> None:
    classes = ["ambulance", "autobus"]
    source_root = tmp_path / "unclean"
    frame = _write_prediction_table(
        source_root,
        [
            ("ambulance", "ambulance", 0.97, False, False),  # kept
            ("autobus", "autobus", 0.95, False, False),  # kept
            ("autobus", "ambulance", 0.99, False, False),  # model disagrees
            ("ambulance", "ambulance", 0.60, True, False),  # low confidence
            ("neysan", "ambulance", 0.98, False, True),  # unseen class
        ],
    )
    kept, rejected = select_confident_known_rows(frame, classes, 0.85)
    assert sorted(kept["source_label"]) == ["ambulance", "autobus"]
    reasons = set(rejected["selection_reason"])
    assert reasons == {
        "unseen_class",
        "model_disagrees_with_label",
        "confidence_below_review_threshold",
    }


def test_export_copy_leaves_source_intact_and_loads_as_imagefolder(
    tmp_path: Path,
) -> None:
    classes = ["ambulance", "autobus"]
    source_root = tmp_path / "unclean"
    destination_root = tmp_path / "cleaned_unclean"
    frame = _write_prediction_table(
        source_root,
        [
            ("ambulance", "ambulance", 0.97, False, False),
            ("ambulance", "ambulance", 0.96, False, False),
            ("autobus", "autobus", 0.95, False, False),
            ("autobus", "ambulance", 0.99, False, False),
        ],
    )
    manifest, _, summary = export_confident_subset(
        frame,
        classes,
        0.85,
        source_root,
        destination_root,
        mode="copy",
    )
    assert summary["rows_selected"] == 3
    assert summary["rows_transferred"] == 3
    assert summary["class_counts"] == {"ambulance": 2, "autobus": 1}
    assert verify_export(manifest, destination_root)["verified"] is True
    # Copy mode must not remove anything from the original split.
    assert len(list(source_root.rglob("*.png"))) == 4
    exported = datasets.ImageFolder(destination_root, transform=transforms.ToTensor())
    assert exported.classes == classes
    assert len(exported) == 3


def test_export_move_removes_selected_files_from_source(tmp_path: Path) -> None:
    source_root = tmp_path / "unclean"
    destination_root = tmp_path / "cleaned_unclean"
    frame = _write_prediction_table(
        source_root,
        [
            ("ambulance", "ambulance", 0.97, False, False),
            ("autobus", "ambulance", 0.99, False, False),
        ],
    )
    manifest, _, summary = export_confident_subset(
        frame,
        ["ambulance", "autobus"],
        0.85,
        source_root,
        destination_root,
        mode="move",
    )
    assert summary["rows_transferred"] == 1
    assert manifest["status"].tolist().count("moved") == 1
    assert [path.name for path in sorted(source_root.rglob("*.png"))] == [
        "autobus_1.png"
    ]
    assert verify_export(manifest, destination_root)["verified"] is True


def test_export_skips_missing_source_without_creating_empty_class(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "unclean"
    destination_root = tmp_path / "cleaned_unclean"
    frame = _write_prediction_table(
        source_root, [("ambulance", "ambulance", 0.97, False, False)]
    )
    next(source_root.rglob("*.png")).unlink()
    manifest, _, summary = export_confident_subset(
        frame,
        ["ambulance", "autobus"],
        0.85,
        source_root,
        destination_root,
        mode="move",
    )
    assert summary["rows_transferred"] == 0
    assert manifest["status"].tolist() == ["missing_source"]
    assert verify_export(manifest, destination_root)["verified"] is True
    assert not destination_root.exists()
