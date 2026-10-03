from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from torchvision import datasets, transforms

from traffic_classifier.audit import _quarantined_split, _write_audit_markdown
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
    count_images,
    make_loader,
    paths_for_subset,
    split_image_count,
)
from traffic_classifier.engine import (
    TrainOptions,
    _compatible_checkpoint,
    _one_hot_if_needed,
    _rolling_checkpoint_path,
    train_model,
)
from traffic_classifier.locking import ProjectRunLock, RunLockError
from traffic_classifier.models import TrafficCNN, build_model, parameter_report
from traffic_classifier.pipeline import experiment_specs, skipped_unclean_result
from traffic_classifier.promotion import (
    BEST_MODEL_NAME,
    load_promoted_checkpoint,
    promote_best_checkpoints,
    read_manifest,
    sha256_file,
    verify_complete,
)
from traffic_classifier.reporting import (
    _leakage_clause,
    _quarantine_sentence,
    render_unclean_analysis,
)
from traffic_classifier.utils import save_checkpoint

from scripts.resolve_split_duplicates import resolve


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


def test_image_count_treats_drained_and_missing_splits_as_empty(tmp_path: Path) -> None:
    # A `--mode move` export drains the split but leaves the class folders
    # behind, so counting directory entries would wrongly report it as
    # populated. Both that state and a missing directory must count as 0.
    assert count_images(tmp_path / "absent") == 0
    assert split_image_count(tmp_path, "unclean") == 0

    source_root = tmp_path / "unclean"
    _write_prediction_table(
        source_root, [("ambulance", "ambulance", 0.97, False, False)]
    )
    assert split_image_count(tmp_path, "unclean") == 1

    for image in source_root.rglob("*.png"):
        image.unlink()
    assert split_image_count(tmp_path, "unclean") == 0
    assert (source_root / "ambulance").is_dir()


def test_skipped_unclean_result_keeps_the_reporting_contract() -> None:
    result = skipped_unclean_result("the unclean split holds no image files")
    assert result["skipped"] is True
    assert result["cleaning"] == {
        "original_images": 0,
        "retained_images": 0,
        "excluded_duplicates": 0,
        "known_images": 0,
        "unseen_neysan_images": 0,
    }
    assert result["known_class_metrics"]["accuracy"] == 0.0
    assert result["known_class_metrics"]["macro_f1"] == 0.0
    assert result["unseen_neysan"]["present"] is False


def test_unclean_report_states_the_skip_instead_of_inventing_numbers() -> None:
    report = render_unclean_analysis(
        skipped_unclean_result("the unclean split holds no image files")
    )
    assert "**skipped**" in report
    assert "the unclean split holds no image files" in report
    # A skipped pass must not present zeros as a measured result.
    assert "0.0000" not in report


def test_unclean_report_still_renders_a_measured_result() -> None:
    report = render_unclean_analysis(
        {
            "skipped": False,
            "cleaning": {
                "original_images": 12,
                "retained_images": 10,
                "excluded_duplicates": 2,
                "known_images": 8,
                "unseen_neysan_images": 2,
            },
            "known_class_metrics": {"accuracy": 0.75, "macro_f1": 0.7},
            "unseen_neysan": {"present": False, "reason": "drained"},
        }
    )
    assert "**10** retained images from 12" in report
    assert "accuracy is **0.7500**" in report
    assert "macro-F1 is **0.7000**" in report


def _split_summary(images: int) -> dict[str, Any]:
    return {
        "images": images,
        "class_counts": {"ambulance": images},
        "width_quantiles": {"min": 100, "max": 100},
        "height_quantiles": {"min": 100, "max": 100},
        "unique_pixel_sha256": images,
    }


def test_audit_markdown_reports_a_split_that_holds_no_images(tmp_path: Path) -> None:
    # An image-less split contributes no group to `summary["splits"]`; the
    # writer must show it as empty instead of raising KeyError.
    config = ProjectConfig(reports_dir=str(tmp_path / "reports"))
    summary: dict[str, Any] = {
        "splits": {"train": _split_summary(4), "test": _split_summary(2)},
        "issue_counts": {},
        "total_images": 6,
        "total_raw_unique_hashes": 6,
        "total_canonical_pixel_unique_hashes": 6,
        "raw_file_duplicate_groups": 0,
        "canonical_duplicate_groups": 0,
        "canonical_duplicate_extra_copies": 0,
        "cross_split_duplicate_groups": 0,
        "label_conflict_duplicate_groups": 0,
        "train_test_overlap_groups": 0,
        "cleaning": {
            "train_excluded": 0,
            "test_excluded": 0,
            "unclean_excluded": 0,
            "split_duplicate_policy": "Keep the frozen-test copy.",
            "quarantined_records": 0,
            "clean_counts": {
                "train": 4,
                "test": 2,
                "unclean": 0,
                "unclean_class_counts": {},
            },
        },
    }
    _write_audit_markdown(config, summary, [], [], [])
    report = (tmp_path / "reports" / "data_audit.md").read_text(encoding="utf-8")
    assert "| unclean | 0 | - | - | - | 0 |" in report


def _duplicate_group(
    train_path: Path, test_path: Path, label: str = "ambulance"
) -> dict[str, Any]:
    return {
        "group": 1,
        "pixel_sha256": "deadbeef",
        "splits": ["test", "train"],
        "label_conflict": False,
        "members": [
            {"split": "train", "label": label, "path": train_path.as_posix()},
            {"split": "test", "label": label, "path": test_path.as_posix()},
        ],
    }


def test_resolve_quarantines_the_train_copy_and_keeps_the_frozen_one(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train" / "ambulance" / "a.jpg"
    test_path = tmp_path / "test" / "ambulance" / "a.jpg"
    moves, unresolved = resolve([_duplicate_group(train_path, test_path)])
    assert unresolved == []
    assert len(moves) == 1
    # The frozen evaluation copy must survive; only the training copy moves.
    assert moves[0]["source_path"] == train_path.as_posix()
    assert moves[0]["kept_duplicate_of"] == test_path.as_posix()
    assert moves[0]["source_split"] == "train"


def test_resolve_leaves_an_ambiguous_group_for_the_audit(tmp_path: Path) -> None:
    # A group spanning an extra split has no safe automatic answer, so it must
    # be handed back to the fail-closed audit rather than guessed at.
    group = _duplicate_group(
        tmp_path / "train" / "ambulance" / "a.jpg",
        tmp_path / "test" / "ambulance" / "a.jpg",
    )
    group["members"].append(
        {"split": "unclean", "label": "neysan", "path": "unclean/neysan/a.jpg"}
    )
    moves, unresolved = resolve([group])
    assert moves == []
    assert len(unresolved) == 1


def test_quarantined_split_reads_the_authoritative_path() -> None:
    # `source_split` is preferred, but the path is authoritative if it is
    # missing or was hand-edited.
    assert (
        _quarantined_split(
            {
                "source_split": "train",
                "source_path": "/data/train/ambulance/a.jpg",
            }
        )
        == "train"
    )
    assert (
        _quarantined_split({"source_path": "/data/test/ambulance/a.jpg"}) == "test"
    )
    assert _quarantined_split({"source_path": "/elsewhere/a.jpg"}) == ""


def test_leakage_narrative_follows_the_recorded_counts() -> None:
    resolved: dict[str, Any] = {
        "canonical_duplicate_groups": 0,
        "label_conflict_duplicate_groups": 0,
        "cleaning": {
            "train_excluded": 8,
            "test_excluded": 0,
            "unclean_excluded": 0,
        },
    }
    assert _leakage_clause(resolved) == ""
    sentence = _quarantine_sentence(resolved)
    assert "8 pixel-identical train copies quarantined" in sentence
    assert "Frozen test" not in sentence

    conflicted: dict[str, Any] = {
        **resolved,
        "canonical_duplicate_groups": 2,
        "label_conflict_duplicate_groups": 1,
    }
    assert _leakage_clause(conflicted) == ", including 1 conflict"
    assert "1 pixel-identical train copy quarantined" in _quarantine_sentence(
        {**resolved, "cleaning": {**resolved["cleaning"], "train_excluded": 1}}
    )


def _fake_experiment_record(
    root: Path, names: list[str], fingerprint: str = "fp-test"
) -> dict[str, Any]:
    """Write minimal checkpoints and return an experiment_results-shaped record."""

    checkpoints = root / "artifacts" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {}
    for index, name in enumerate(names):
        payload = {
            "model_state": {"weight": torch.tensor([float(index)])},
            "metadata": {"run_fingerprint": fingerprint},
        }
        path = checkpoints / f"{name}.pt"
        save_checkpoint(path, payload)
        record[name] = {
            "checkpoint_path": str(path),
            "architecture": "cnn",
            "category": "main",
            "loss": "cross_entropy",
            "best_epoch": index + 1,
            "validation_metrics": {"macro_f1": 0.5 + index / 100.0},
            "metadata": {"run_fingerprint": fingerprint},
        }
    return record


def test_promotion_copies_winner_and_records_hashes(tmp_path: Path) -> None:
    config = ProjectConfig(artifacts_dir=str(tmp_path / "artifacts"))
    names = ["alpha", "beta", "gamma"]
    record = _fake_experiment_record(tmp_path, names)

    manifest = promote_best_checkpoints(
        config=config,
        experiment_results=record,
        selected_name="beta",
        run_fingerprint="fp-test",
        expected=set(names),
        paths_root=tmp_path,
    )

    directory = tmp_path / "artifacts" / "best_checkpoints"
    assert manifest["selected_experiment"] == "beta"
    # the winner has the highest validation macro-F1
    assert manifest["production_validation_macro_f1"] == 0.51
    assert (directory / BEST_MODEL_NAME).is_file()
    for name in names:
        assert (directory / f"{name}.pt").is_file()

    source = Path(record["beta"]["checkpoint_path"])
    assert sha256_file(directory / BEST_MODEL_NAME) == sha256_file(source)
    assert manifest["production_checkpoint_sha256"] == sha256_file(source)

    # best_model.pt must load, and the manifest round-trips
    loaded = load_promoted_checkpoint(config, manifest=manifest)
    assert torch.equal(loaded["model_state"]["weight"], torch.tensor([1.0]))
    assert read_manifest(config)["selected_experiment"] == "beta"


def test_promotion_refuses_a_partial_experiment_record(tmp_path: Path) -> None:
    config = ProjectConfig(artifacts_dir=str(tmp_path / "artifacts"))
    names = ["alpha", "beta", "gamma"]
    record = _fake_experiment_record(tmp_path, names)
    # a run interrupted mid-write leaves only some arms behind; promoting that
    # subset could crown an experiment that merely lost to a missing arm
    del record["gamma"]

    with pytest.raises(RuntimeError, match="incomplete experiment record"):
        promote_best_checkpoints(
            config=config,
            experiment_results=record,
            selected_name="beta",
            run_fingerprint="fp-test",
            expected=set(names),
            paths_root=tmp_path,
        )
    assert not (tmp_path / "artifacts" / "best_checkpoints" / BEST_MODEL_NAME).exists()


def test_promotion_refuses_checkpoints_from_another_run(tmp_path: Path) -> None:
    config = ProjectConfig(artifacts_dir=str(tmp_path / "artifacts"))
    names = ["alpha", "beta"]
    record = _fake_experiment_record(tmp_path, names)
    record["beta"]["metadata"]["run_fingerprint"] = "fp-older"

    with pytest.raises(RuntimeError, match="different run"):
        promote_best_checkpoints(
            config=config,
            experiment_results=record,
            selected_name="beta",
            run_fingerprint="fp-test",
            expected=set(names),
            paths_root=tmp_path,
        )


def test_promotion_drops_checkpoints_of_removed_experiments(tmp_path: Path) -> None:
    config = ProjectConfig(artifacts_dir=str(tmp_path / "artifacts"))
    record = _fake_experiment_record(tmp_path, ["alpha", "beta"])
    promote_best_checkpoints(
        config=config,
        experiment_results=record,
        selected_name="beta",
        run_fingerprint="fp-test",
        expected={"alpha", "beta"},
        paths_root=tmp_path,
    )
    directory = tmp_path / "artifacts" / "best_checkpoints"
    # simulate a leftover file from a superseded run
    (directory / "retired_arm.pt").write_bytes(b"stale")

    reduced = {name: record[name] for name in ("beta",)}
    (tmp_path / "artifacts" / "checkpoints" / "alpha.pt").unlink()
    promote_best_checkpoints(
        config=config,
        experiment_results=reduced,
        selected_name="beta",
        run_fingerprint="fp-test",
        expected={"beta"},
        paths_root=tmp_path,
    )

    assert not (directory / "retired_arm.pt").exists()
    assert not (directory / "alpha.pt").exists()
    assert (directory / "beta.pt").is_file()


def test_loading_a_modified_promoted_checkpoint_fails_loudly(tmp_path: Path) -> None:
    config = ProjectConfig(artifacts_dir=str(tmp_path / "artifacts"))
    record = _fake_experiment_record(tmp_path, ["alpha"])
    promote_best_checkpoints(
        config=config,
        experiment_results=record,
        selected_name="alpha",
        run_fingerprint="fp-test",
        expected={"alpha"},
        paths_root=tmp_path,
    )

    (tmp_path / "artifacts" / "best_checkpoints" / BEST_MODEL_NAME).write_bytes(
        b"tampered"
    )
    with pytest.raises(RuntimeError, match="was modified after promotion"):
        load_promoted_checkpoint(config)


def test_verify_complete_defaults_to_the_declared_experiment_specs() -> None:
    declared = {spec.name for spec in experiment_specs()}
    complete = {
        name: {"metadata": {"run_fingerprint": "fp"}} for name in declared
    }
    verify_complete(complete, run_fingerprint="fp")

    incomplete = dict(complete)
    incomplete.pop("resnet18_fine_tuning")
    with pytest.raises(RuntimeError, match="resnet18_fine_tuning"):
        verify_complete(incomplete, run_fingerprint="fp")


def test_training_end_prints_the_chosen_best_checkpoint(
    tmp_path: Path, capsys: Any
) -> None:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    root = tmp_path / "data"
    for label in ("alpha", "beta"):
        class_dir = root / label
        class_dir.mkdir(parents=True)
        for index in range(4):
            Image.fromarray(pixels + index * 10).save(class_dir / f"{index}.png")

    dataset = datasets.ImageFolder(root, transform=transforms.ToTensor())
    classes = list(dataset.classes)
    train_loader = make_loader(
        dataset, range(len(dataset)), training=True, batch_size=2, seed=1
    )
    validation_loader = make_loader(
        dataset, range(len(dataset)), training=False, batch_size=2, seed=1
    )
    options = TrainOptions(
        experiment_name="demo",
        architecture="cnn",
        epochs=2,
        resume=False,
        checkpoint_path=str(tmp_path / "demo.pt"),
    )
    model = build_model("cnn", len(classes))

    capsys.readouterr()
    train_model(
        model,
        train_loader,
        validation_loader,
        options,
        class_names=classes,
        device=torch.device("cpu"),
    )
    output = capsys.readouterr().out

    assert "training end: best checkpoint chosen at epoch" in output
    assert "by validation macro-F1" in output
    assert f"best checkpoint: {(tmp_path / 'demo.pt').as_posix()}" in output


def test_reusing_a_completed_run_still_prints_the_chosen_checkpoint(
    tmp_path: Path, capsys: Any
) -> None:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    root = tmp_path / "data"
    for label in ("alpha", "beta"):
        class_dir = root / label
        class_dir.mkdir(parents=True)
        for index in range(4):
            Image.fromarray(pixels + index * 10).save(class_dir / f"{index}.png")

    dataset = datasets.ImageFolder(root, transform=transforms.ToTensor())
    classes = list(dataset.classes)
    loader = make_loader(
        dataset, range(len(dataset)), training=True, batch_size=2, seed=1
    )
    validation_loader = make_loader(
        dataset, range(len(dataset)), training=False, batch_size=2, seed=1
    )
    checkpoint = tmp_path / "demo.pt"
    options = TrainOptions(
        experiment_name="demo",
        architecture="cnn",
        epochs=2,
        resume=True,
        checkpoint_path=str(checkpoint),
    )
    model = build_model("cnn", len(classes))

    train_model(
        model, loader, validation_loader, options,
        class_names=classes, device=torch.device("cpu"),
    )
    assert checkpoint.exists()

    # A second call resumes the completed checkpoint and must still announce it.
    capsys.readouterr()
    train_model(
        model, loader, validation_loader, options,
        class_names=classes, device=torch.device("cpu"),
    )
    output = capsys.readouterr().out

    assert "reusing completed checkpoint" in output
    assert "training end: best checkpoint chosen at epoch" in output
    assert f"best checkpoint: {checkpoint.as_posix()}" in output
