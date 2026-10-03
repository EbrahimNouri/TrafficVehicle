"""Post-hoc diagnostic: score every trained experiment on the frozen test set.

Read-only with respect to model selection. This script runs *after* the
validation-selected production model and its single official frozen-test
evaluation are already locked in
``artifacts/results/frozen_test_evaluation.json``.

It exists for analysis and reporting only:

* it does not participate in model selection, hyperparameter choice,
  temperature fitting, or review-threshold selection;
* it never writes ``frozen_test_evaluation.json``, ``production_ready.json``,
  the final checkpoint, or any file under ``reports/``;
* the official frozen-test result remains the single evaluation recorded in
  ``frozen_test_evaluation.json``.

Because these are per-model test scores, they must not be used to pick a
different production model. Any such change would turn the test set into a
selection tool and invalidate the project's leakage guarantees.

Every printed line is mirrored to
``artifacts/logs/all_models_test_diagnostic.log``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, TextIO

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_classifier.config import ProjectConfig  # noqa: E402
from traffic_classifier.data import (  # noqa: E402
    build_transforms,
    load_image_folder,
    make_loader,
)
from traffic_classifier.engine import evaluate  # noqa: E402
from traffic_classifier.metrics import (  # noqa: E402
    classification_metrics,
    confusion_pair_ranking,
)
from traffic_classifier.models import build_model  # noqa: E402
from traffic_classifier.utils import load_checkpoint  # noqa: E402

DIAGNOSTIC_JSON = "all_models_test_diagnostic.json"
DIAGNOSTIC_CSV = "all_models_test_diagnostic.csv"
LOG_PATH = (
    ROOT / "artifacts" / "logs" / "all_models_test_diagnostic.log"
)

BANNER = """\
================================================================================
POST-HOC DIAGNOSTIC - NOT A SELECTION STEP
================================================================================
Every trained experiment below is scored on dataset/test for analysis only.
The production model was already chosen on validation macro-F1, and its single
official test evaluation is already recorded and locked. Nothing printed here
changes that decision. Do not use these numbers to choose a different model.
================================================================================
"""


class _Tee:
    """Mirror stdout into a log file so the whole run is captured."""

    def __init__(self, stream: TextIO, path: Path) -> None:
        self._stream = stream
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8")

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._handle.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


def _banner(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def _check_transform(expected: dict[str, Any], actual: Any, name: str) -> None:
    """Fail loudly if a rebuilt transform differs from the one used in training."""

    if int(expected["image_size"]) != int(actual.image_size):
        raise RuntimeError(
            f"{name}: evaluation image size {actual.image_size} does not match the "
            f"stored {expected['image_size']}"
        )
    for field in ("normalization_mean", "normalization_std"):
        stored = [float(value) for value in expected[field]]
        rebuilt = [float(value) for value in getattr(actual, field)]
        if len(stored) != len(rebuilt) or any(
            abs(a - b) > 1e-9 for a, b in zip(stored, rebuilt)
        ):
            raise RuntimeError(
                f"{name}: rebuilt {field} {rebuilt} does not match stored {stored}"
            )


def run(config_path: Path) -> dict[str, Any]:
    config = ProjectConfig.from_json(config_path)
    results_dir = f"{ROOT}\\artifacts\\results"
    device = config.resolve_device()

    print(BANNER)
    print(f"config        : {config_path}")
    print(f"artifacts dir : {config.artifacts_dir}")
    print(f"device        : {device}")

    # --- guard: the official single test evaluation must already exist --------
    frozen_path = Path(f"{results_dir}\\frozen_test_evaluation.json")
    if not frozen_path.exists():
        raise SystemExit(
            "refusing to run: no official frozen-test evaluation exists yet.\n"
            "Run `python main.py` first so the production model is selected on "
            "validation and evaluated once, then run this diagnostic."
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    protocol = frozen["evaluation_protocol"]
    official_name = frozen["selected_experiment"]
    official_metrics = frozen["test_metrics"]

    _banner("Locked official result (authoritative, not recomputed here)")
    print(f"selected on        : {protocol['selected_using']}")
    print(f"production model   : {official_name}")
    print(f"test evaluations   : {protocol['test_evaluations']}")
    print(f"official test acc  : {official_metrics['accuracy']:.4f}")
    print(f"official test F1   : {official_metrics['macro_f1']:.4f}")
    print(f"temperature        : {frozen['temperature']:.4f}")
    print(f"review threshold   : {frozen['review_threshold']['threshold']:.4f}")

    experiments_path = Path(f"{results_dir}\\experiment_results.json")
    if not experiments_path.exists():
        raise SystemExit(f"missing experiment record: {experiments_path}")
    experiments = json.loads(experiments_path.read_text(encoding="utf-8"))

    # every experiment must agree on the class mapping before we reuse it
    mappings = {
        name: tuple(sorted(entry["metadata"]["class_mapping"].items()))
        for name, entry in experiments.items()
    }
    if len(set(mappings.values())) != 1:
        raise RuntimeError("experiments disagree on the class mapping")
    print(f"\nclass mapping verified identical across {len(experiments)} experiments")

    dataset_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for name in sorted(experiments):
        entry = experiments[name]
        meta = entry["metadata"]
        stored_transform = meta["evaluation_transform"]
        kind = stored_transform["name"]

        transform, spec = build_transforms(
            config, augmentation=False, kind=kind  # type: ignore[arg-type]
        )
        _check_transform(stored_transform, spec, name)

        if kind not in dataset_cache:
            dataset_cache[kind] = load_image_folder(
                f"{ROOT}\\dataset", "test", transform, config.classes
            )
        dataset = dataset_cache[kind]
        loader = make_loader(
            dataset,
            range(len(dataset)),
            training=False,
            batch_size=config.batch_size,
            seed=config.seed,
            num_workers=config.num_workers,
        )

        checkpoint_path = (
            Path(f"{ROOT}\\artifacts") / "checkpoints" / f"{name}.pt"
        )
        if not checkpoint_path.is_file():
            raise SystemExit(f"missing checkpoint for {name}: {checkpoint_path}")

        model = build_model(
            entry["architecture"],
            len(config.classes),
            dropout=float(meta["dropout"]),
            pooling=meta["pooling"],  # type: ignore[arg-type]
            pretrained=False,
        )
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)

        is_bce = entry["loss"] == "bce"
        result = evaluate(
            model,
            loader,
            nn.BCEWithLogitsLoss() if is_bce else nn.CrossEntropyLoss(),
            device,
            loss_name=entry["loss"],
            num_classes=len(config.classes),
            class_names=config.classes,
            probability_kind="sigmoid" if is_bce else "softmax",
        )
        outputs = result["outputs"]
        targets = outputs["targets"]
        predictions = outputs["predictions"]
        metrics = classification_metrics(targets, predictions, config.classes)
        pairs = confusion_pair_ranking(
            metrics["confusion_matrix_row_normalized"], config.classes
        )

        rows.append(
            {
                "experiment": name,
                "display_name": entry["display_name"],
                "category": entry["category"],
                "architecture": entry["architecture"],
                "loss": entry["loss"],
                "best_epoch": entry["best_epoch"],
                "validation_macro_f1": float(
                    entry["validation_metrics"]["macro_f1"]
                ),
                "test_accuracy": float(metrics["accuracy"]),
                "test_macro_precision": float(metrics["macro_precision"]),
                "test_macro_recall": float(metrics["macro_recall"]),
                "test_macro_f1": float(metrics["macro_f1"]),
                "test_errors": int((targets != predictions).sum()),
                "test_images": int(len(targets)),
                "top_confusion_pair": (
                    f"{pairs[0]['class_a']}/{pairs[0]['class_b']} "
                    f"({pairs[0]['pair_confusion']:.3f})"
                    if pairs
                    else ""
                ),
                "top_confusion_a_as_b": float(pairs[0]["a_as_b"]) if pairs else 0.0,
                "top_confusion_b_as_a": float(pairs[0]["b_as_a"]) if pairs else 0.0,
                "is_official_production_model": name == official_name,
            }
        )
        print(
            f"evaluated {name:<28} test macro-F1 {rows[-1]['test_macro_f1']:.4f} "
            f"acc {rows[-1]['test_accuracy']:.4f} "
            f"errors {rows[-1]['test_errors']}/{rows[-1]['test_images']}"
        )

    # --- integrity check: our recomputation must match the official record ----
    official_row = next(
        (row for row in rows if row["experiment"] == official_name), None
    )
    if official_row is None:
        raise RuntimeError(
            f"production model {official_name} is absent from the experiment record"
        )
    drift = abs(
        official_row["test_macro_f1"] - float(official_metrics["macro_f1"])
    )
    if drift > 1e-6:
        raise RuntimeError(
            "recomputed test macro-F1 for the production model disagrees with the "
            f"locked record by {drift:.3e}; refusing to publish this diagnostic"
        )
    print(
        f"\nintegrity check passed: recomputed {official_name} test macro-F1 "
        f"{official_row['test_macro_f1']:.6f} matches the locked "
        f"{float(official_metrics['macro_f1']):.6f}"
    )

    rows.sort(key=lambda row: -row["test_macro_f1"])

    _banner("All experiments ranked by test macro-F1 (diagnostic only)")
    header = (
        f"{'test F1':>8}  {'test acc':>8}  {'val F1':>7}  {'err':>4}  "
        f"{'ep':>3}  experiment"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        marker = " *" if row["is_official_production_model"] else "  "
        print(
            f"{row['test_macro_f1']:>8.4f}  {row['test_accuracy']:>8.4f}  "
            f"{row['validation_macro_f1']:>7.4f}  "
            f"{row['test_errors']:>4}  {row['best_epoch']:>3}  "
            f"{row['experiment']}{marker}"
        )
    print("\n* = official production model (selected on validation, test used once)")

    imbalance = [row for row in rows if row["category"] == "imbalance"]
    if imbalance:
        _banner("Note on the imbalance arms")
        print(
            "The two imbalance arms were trained on a deliberately reduced subset "
            "(14 images for kamyun/minibus/savari/vanet) to exercise the "
            "BalancedBatchSampler. Their test scores are not comparable to the "
            "full-data arms and cannot be production candidates."
        )

    _banner("Residual error of the production model")
    print(f"top confusion pair: {official_row['top_confusion_pair']}")

    payload = {
        "purpose": (
            "post-hoc diagnostic; not used for model selection, hyperparameters, "
            "temperature, or review threshold"
        ),
        "test_split": {
            "source": "dataset/test",
            "images": int(official_row["test_images"]),
            "per_class": 50,
            "used_for_official_evaluation_count": protocol["test_evaluations"],
        },
        "official_production_model": official_name,
        "official_test_macro_f1": float(official_metrics["macro_f1"]),
        "integrity_check": {
            "recomputed_macro_f1": official_row["test_macro_f1"],
            "locked_macro_f1": float(official_metrics["macro_f1"]),
            "abs_difference": drift,
            "passed": True,
        },
        "results": rows,
    }
    json_path = results_dir / DIAGNOSTIC_JSON
    csv_path = results_dir / DIAGNOSTIC_CSV
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _banner("Written")
    print(f"diagnostic json : {json_path.as_posix()}")
    print(f"diagnostic csv  : {csv_path.as_posix()}")
    print(f"log             : {LOG_PATH.as_posix()}")
    print(
        "\nReminder: the official result is unchanged and remains the single "
        "frozen-test evaluation recorded in frozen_test_evaluation.json."
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc diagnostic: score every trained experiment on dataset/test. "
            "Runs only after the official single frozen-test evaluation exists."
        )
    )
    parser.add_argument("--config", default=f"{ROOT}\\configs\\default.json")
    args = parser.parse_args()

    tee = _Tee(sys.__stdout__, LOG_PATH)
    sys.stdout = tee
    try:
        run(Path(args.config))
    finally:
        sys.stdout = tee._stream
        tee.close()


if __name__ == "__main__":
    main()