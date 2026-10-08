"""Build a PDF report of every model's validation and test evaluation.

Run it after the pipeline (and, ideally, after the test diagnostic):

    python scripts/generate_pdf_report.py
    python scripts/generate_pdf_report.py --output reports/evaluation_report.pdf

Inputs (all read-only):
    artifacts/results/experiment_results.json        validation metrics, histories
    artifacts/results/all_models_test/summary.json   test metrics for all models
    artifacts/results/all_models_test/per_model/*.csv  per-image test predictions
    artifacts/results/frozen_test_evaluation.json    official test evaluation
    artifacts/results/validation_uncertainty.json    calibration (production model)
    artifacts/production_ready.json                  selected production model

The test split is diagnostic only: the report states that model selection was
performed on the validation split alone.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_hex as _mpl_to_hex  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

EXPERIMENT_ORDER = [
    "cnn_baseline",
    "no_augmentation",
    "dropout_0_3",
    "dropout_0_5",
    "avg_pool",
    "weight_decay_1e-4",
    "step_lr",
    "bce_loss",
    "imbalanced_standard",
    "imbalanced_balanced",
    "best_regularized",
    "resnet18_feature_extraction",
    "resnet18_fine_tuning",
]

PAGE = landscape(A4)
MARGIN = 16 * mm
HEADER_BG = colors.HexColor("#1f2937")
ROW_ALT = colors.HexColor("#f8fafc")
HIGHLIGHT_BG = colors.HexColor("#fde68a")
GRID = colors.HexColor("#d1d5db")
ACCENT = colors.HexColor("#2563eb")


# --------------------------------------------------------------------------- helpers


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC).astimezone()


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load_test_per_class(per_model_dir: Path, class_names: list[str]) -> dict[str, dict[str, float]]:
    """Recompute per-class test F1 for every model from its per-image CSV."""

    per_class: dict[str, dict[str, float]] = {}
    if not per_model_dir.is_dir():
        return per_class
    for path in sorted(per_model_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            frame["true_class"],
            frame["predicted_class"],
            labels=class_names,
            zero_division=0,
        )
        per_class[path.stem] = {
            name: float(score) for name, score in zip(class_names, f1)
        }
    return per_class


def _model_colors() -> dict[str, str]:
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    return {
        name: _mpl_to_hex(palette[index % len(palette)])
        for index, name in enumerate(EXPERIMENT_ORDER)
    }


# ------------------------------------------------------------------------- figures


def _chart_val_vs_test(rows: list[dict[str, Any]], destination: Path) -> Path:
    labels = [row["name"] for row in rows]
    val = [row["val_f1"] for row in rows]
    test = [row["test_f1"] if row["test_f1"] is not None else 0.0 for row in rows]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10.5, 4.4))
    axis.bar(positions - 0.2, val, width=0.4, label="validation macro-F1", color="#2563eb")
    axis.bar(positions + 0.2, test, width=0.4, label="test macro-F1 (diagnostic)", color="#f59e0b")
    for index, row in enumerate(rows):
        if row["official"]:
            axis.annotate(
                "production",
                xy=(index - 0.2, row["val_f1"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#b45309",
            )
    axis.set_xticks(positions, labels, rotation=38, ha="right", fontsize=8)
    axis.set_ylabel("macro-F1")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(fontsize=8, loc="lower right")
    axis.set_title("Validation vs. test macro-F1 for all 13 models", fontsize=11)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return destination


def _chart_val_curves(results: dict[str, Any], destination: Path) -> Path:
    colors_by_name = _model_colors()
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    for name in EXPERIMENT_ORDER:
        history = results.get(name, {}).get("history") or []
        if not history:
            continue
        epochs = [item["epoch"] for item in history]
        f1 = [item["validation_macro_f1"] for item in history]
        axis.plot(epochs, f1, linewidth=1.1, color=colors_by_name[name], label=name)
        best = results[name]["best_epoch"]
        axis.plot(
            [best],
            [results[name]["validation_metrics"]["macro_f1"]],
            marker="o",
            markersize=4,
            color=colors_by_name[name],
        )
    axis.set_xlabel("epoch")
    axis.set_ylabel("validation macro-F1")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_title("Validation macro-F1 per epoch (dot = best epoch)", fontsize=11)
    axis.legend(fontsize=6.5, ncol=2, loc="lower right", framealpha=0.9)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return destination


def _chart_test_per_class_heatmap(
    per_class: dict[str, dict[str, float]],
    test_f1: dict[str, float],
    class_names: list[str],
    destination: Path,
) -> Path:
    ordered = sorted(
        (name for name in per_class if name in test_f1),
        key=lambda name: test_f1[name],
        reverse=True,
    )
    matrix = np.asarray([[per_class[name][cls] for cls in class_names] for name in ordered])
    figure, axis = plt.subplots(figsize=(10.0, 5.6))
    draw = axis.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=38, ha="right")
    axis.set_yticks(range(len(ordered)), labels=ordered)
    axis.set_title("Per-class test F1 (64 test images, diagnostic)", fontsize=11)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if value >= 0.6 else "#7f1d1d",
            )
    figure.colorbar(draw, ax=axis, fraction=0.03, pad=0.02)
    figure.tight_layout()
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return destination


def _draw_confusion_panels(
    panels: list[tuple[str, dict[str, Any]]],
    class_names: list[str],
    destination: Path,
) -> Path:
    """Row-normalized confusion heatmaps with raw counts, one panel per split."""

    columns = len(panels)
    figure, axes = plt.subplots(1, columns, figsize=(6.6 * columns, 5.6), squeeze=False)
    for axis, (label, metrics) in zip(axes[0], panels):
        counts = np.asarray(metrics["confusion_matrix_counts"], dtype=float)
        normalized = np.asarray(metrics["confusion_matrix_row_normalized"], dtype=float)
        draw = axis.imshow(normalized, cmap="Oranges", vmin=0, vmax=1)
        axis.set_xticks(range(len(class_names)), labels=class_names, rotation=38, ha="right")
        axis.set_yticks(range(len(class_names)), labels=class_names)
        axis.set(xlabel="predicted class", ylabel="true class", title=label)
        for row in range(counts.shape[0]):
            for column in range(counts.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{normalized[row, column]:.2f}\n({int(counts[row, column])})",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black" if normalized[row, column] < 0.6 else "white",
                )
        figure.colorbar(draw, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return destination


# ------------------------------------------------------------------------ pdf parts


def _table(
    header: list[str],
    rows: list[list[str]],
    widths: list[float],
    *,
    right_cols: tuple[int, ...] = (),
    center_cols: tuple[int, ...] = (),
    highlight_row: int | None = None,
    font_size: float = 7.5,
) -> Table:
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
    ]
    for column in right_cols:
        style.append(("ALIGN", (column, 0), (column, -1), "RIGHT"))
    for column in center_cols:
        style.append(("ALIGN", (column, 0), (column, -1), "CENTER"))
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), HIGHLIGHT_BG))
        style.append(("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"))
    table = Table([header, *rows], colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle(style))
    return table


def _image(path: Path, max_width: float, max_height: float) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as handle:
        width, height = handle.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(MARGIN, 10 * mm, "Traffic Vehicle Classification - model evaluation report")
    canvas.drawRightString(PAGE[0] - MARGIN, 10 * mm, f"page {doc.page}")
    canvas.restoreState()


# --------------------------------------------------------------------------- report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PDF report of validation and test results for every model."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="reports/evaluation_report.pdf")
    args = parser.parse_args()

    config = _read_json(ROOT / args.config)
    artifacts = ROOT / config.get("artifacts_dir", "artifacts")
    results_dir = artifacts / "results"
    class_names: list[str] = list(config["classes"])

    results_path = results_dir / "experiment_results.json"
    if not results_path.is_file():
        raise SystemExit(f"{results_path} not found; run `python main.py` first")
    results: dict[str, Any] = _read_json(results_path)

    test_summary_path = results_dir / "all_models_test" / "summary.json"
    test_summary = _read_json(test_summary_path) if test_summary_path.is_file() else None
    frozen_path = results_dir / "frozen_test_evaluation.json"
    frozen = _read_json(frozen_path) if frozen_path.is_file() else None
    calibration_path = results_dir / "validation_uncertainty.json"
    calibration = _read_json(calibration_path) if calibration_path.is_file() else None
    production_path = artifacts / "production_ready.json"
    production = _read_json(production_path) if production_path.is_file() else None

    test_rows = {row["experiment"]: row for row in test_summary["summary"]} if test_summary else {}
    official_name = (
        (production or {}).get("selected_experiment")
        or (frozen or {}).get("selected_experiment")
        or (test_summary or {}).get("official_evaluation", {}).get("selected_experiment")
        or "resnet18_fine_tuning"
    )
    test_per_class = _load_test_per_class(results_dir / "all_models_test" / "per_model", class_names)

    sample = next(iter(results.values()))
    metadata = sample.get("metadata", {})
    dataset = metadata.get("dataset", {})
    train_size = len(dataset.get("train_indices", []))
    validation_size = len(dataset.get("validation_indices", []))
    test_size = (
        test_summary["coverage"]["test_images"]
        if test_summary
        else sum(
            (frozen or {}).get("test_metrics", {}).get("per_class", {}).get(cls, {}).get("support", 0)
            for cls in class_names
        )
        or None
    )

    # --- joined per-model rows ------------------------------------------------
    joined: list[dict[str, Any]] = []
    for name in EXPERIMENT_ORDER:
        entry = results.get(name)
        if entry is None:
            continue
        test_row = test_rows.get(name)
        joined.append(
            {
                "name": name,
                "display": entry["display_name"],
                "category": entry["category"],
                "architecture": entry["architecture"],
                "loss": entry["loss"],
                "best_epoch": entry["best_epoch"],
                "val_metrics": entry["validation_metrics"],
                "val_uncertainty": entry.get("validation_uncertainty", {}),
                "elapsed": entry.get("elapsed_seconds"),
                "history": entry.get("history", []),
                "val_f1": entry["validation_metrics"]["macro_f1"],
                "test_f1": test_row["test_macro_f1"] if test_row else None,
                "test": test_row,
                "official": name == official_name,
            }
        )
    by_val = sorted(joined, key=lambda row: row["val_f1"], reverse=True)
    by_test = sorted(
        (row for row in joined if row["test_f1"] is not None),
        key=lambda row: row["test_f1"],
        reverse=True,
    )

    figures = Path(tempfile.mkdtemp(prefix="pdf_report_figures_"))
    try:
        _build(
            args=args,
            config=config,
            class_names=class_names,
            joined=joined,
            by_val=by_val,
            by_test=by_test,
            results=results,
            test_summary=test_summary,
            frozen=frozen,
            calibration=calibration,
            production=production,
            official_name=official_name,
            test_per_class=test_per_class,
            sizes={
                "train": train_size,
                "validation": validation_size,
                "test": test_size,
            },
            figures=figures,
        )
    finally:
        import shutil

        shutil.rmtree(figures, ignore_errors=True)


def _build(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    class_names: list[str],
    joined: list[dict[str, Any]],
    by_val: list[dict[str, Any]],
    by_test: list[dict[str, Any]],
    results: dict[str, Any],
    test_summary: dict[str, Any] | None,
    frozen: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
    production: dict[str, Any] | None,
    official_name: str,
    test_per_class: dict[str, dict[str, float]],
    sizes: dict[str, Any],
    figures: Path,
) -> None:
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=24, leading=28, textColor=HEADER_BG
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4b5563"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=HEADER_BG,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody", parent=styles["Normal"], fontSize=9, leading=13, spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=body_style,
        leftIndent=12,
        bulletIndent=2,
        spaceAfter=3,
    )
    note_style = ParagraphStyle(
        "ReportNote",
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6b7280"),
    )

    story: list[Any] = []
    page_width = PAGE[0] - 2 * MARGIN

    # --- cover ---------------------------------------------------------------
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("Traffic Vehicle Classification", title_style))
    story.append(
        Paragraph(
            "Model evaluation report - validation and test performance of all "
            f"{len(joined)} trained models",
            subtitle_style,
        )
    )

    official = next((row for row in joined if row["official"]), None)
    fingerprint = (
        (production or {}).get("run_fingerprint")
        or metadata_fingerprint(results)
        or "-"
    )
    cover_rows = [
        ["generated", _now().strftime("%Y-%m-%d %H:%M:%S")],
        ["config", args.config],
        ["run fingerprint", fingerprint],
        ["classes", f"{len(class_names)}: {', '.join(class_names)}"],
        [
            "dataset split",
            f"{sizes['train']} train / {sizes['validation']} validation"
            + (f" / {sizes['test']} test" if sizes["test"] else ""),
        ],
        ["models", f"{len(joined)} experiments, all {config.get('epochs', '-')} epochs"],
        [
            "production model",
            f"{official['display']} ({official_name})" if official else "not selected",
        ],
        [
            "production checkpoint sha256",
            (production or {}).get("final_checkpoint_sha256", "-"),
        ],
        ["selection protocol", "best validation macro-F1; test split used for reporting only"],
    ]
    story.append(
        _table(
            ["field", "value"],
            cover_rows,
            [55 * mm, page_width - 55 * mm],
            font_size=8.5,
        )
    )
    story.append(Spacer(1, 6 * mm))

    # --- key findings --------------------------------------------------------
    story.append(Paragraph("Key findings", heading_style))
    findings = _findings(
        joined=joined,
        by_val=by_val,
        by_test=by_test,
        official=official,
        frozen=frozen,
        test_summary=test_summary,
    )
    for item in findings:
        story.append(Paragraph(item, bullet_style, bulletText="\u2022"))
    story.append(PageBreak())

    # --- overview ------------------------------------------------------------
    story.append(Paragraph("1. Results overview (validation and test)", heading_style))
    if by_test:
        story.append(
            Paragraph(
                "Sorted by validation macro-F1. Test columns come from the "
                "post-selection diagnostic over the 64-image test split.",
                note_style,
            )
        )
    else:
        story.append(
            Paragraph(
                "Test results are missing: run "
                "<font face='Courier'>python scripts/evaluate_all_models_on_test.py</font> "
                "to populate the test columns.",
                note_style,
            )
        )
    story.append(Spacer(1, 3))

    overview_header = [
        "model",
        "category",
        "arch",
        "epoch",
        "val acc",
        "val macro-F1",
        "test acc",
        "test macro-F1",
        "test errors",
        "test mean conf.",
    ]
    overview_rows: list[list[str]] = []
    official_index: int | None = None
    for row in by_val:
        test = row["test"] or {}
        if row["official"]:
            official_index = 1 + len(overview_rows)
        overview_rows.append(
            [
                row["display"],
                row["category"],
                row["architecture"],
                str(row["best_epoch"]),
                _fmt(row["val_metrics"]["accuracy"]),
                _fmt(row["val_f1"]),
                _fmt(test.get("test_accuracy")),
                _fmt(row["test_f1"]),
                (
                    f"{test['test_errors']}/{test['test_images']}"
                    if test
                    else "-"
                ),
                _fmt(test.get("mean_confidence")),
            ]
        )
    story.append(
        _table(
            overview_header,
            overview_rows,
            [52 * mm, 20 * mm, 17 * mm, 14 * mm, 19 * mm, 24 * mm, 19 * mm, 24 * mm, 20 * mm, 24 * mm],
            right_cols=(3, 4, 5, 6, 7, 8, 9),
            highlight_row=official_index,
        )
    )
    story.append(Spacer(1, 5 * mm))

    overview_chart = _chart_val_vs_test(by_val, figures / "val_vs_test.png")
    story.append(_image(overview_chart, page_width, 60 * mm))
    story.append(PageBreak())

    # --- validation ----------------------------------------------------------
    story.append(Paragraph("2. Validation evaluation (232 images, all models)", heading_style))
    story.append(
        Paragraph(
            "Metrics at each model's best epoch, selected on the validation macro-F1. "
            "ECE is the 10-bin expected calibration error of the raw (uncalibrated) "
            "validation probabilities.",
            note_style,
        )
    )
    story.append(Spacer(1, 3))
    val_header = [
        "model",
        "loss",
        "epoch",
        "acc",
        "macro-P",
        "macro-R",
        "macro-F1",
        "weighted-F1",
        "ECE",
        "mean conf.",
        "elapsed s",
    ]
    val_rows: list[list[str]] = []
    official_index = None
    for row in by_val:
        metrics = row["val_metrics"]
        uncertainty = row["val_uncertainty"]
        if row["official"]:
            official_index = 1 + len(val_rows)
        val_rows.append(
            [
                row["display"],
                row["loss"],
                str(row["best_epoch"]),
                _fmt(metrics["accuracy"]),
                _fmt(metrics["macro_precision"]),
                _fmt(metrics["macro_recall"]),
                _fmt(metrics["macro_f1"]),
                _fmt(metrics["weighted_f1"]),
                _fmt(uncertainty.get("ece_argmax_10_bins")),
                _fmt(uncertainty.get("mean_confidence")),
                f"{row['elapsed']:.0f}" if row["elapsed"] else "-",
            ]
        )
    story.append(
        _table(
            val_header,
            val_rows,
            [50 * mm, 24 * mm, 14 * mm, 17 * mm, 18 * mm, 18 * mm, 21 * mm, 23 * mm, 16 * mm, 21 * mm, 19 * mm],
            right_cols=(2, 3, 4, 5, 6, 7, 8, 9, 10),
            highlight_row=official_index,
        )
    )
    story.append(Spacer(1, 5 * mm))

    curves_chart = _chart_val_curves(results, figures / "val_curves.png")
    story.append(_image(curves_chart, page_width, 60 * mm))
    story.append(PageBreak())

    # --- test ----------------------------------------------------------------
    story.append(Paragraph("3. Test evaluation (64 images, diagnostic)", heading_style))
    if by_test:
        story.append(
            Paragraph(
                "The test split was never used for model selection or tuning; these "
                "numbers are a post-selection diagnostic. Ranked by test macro-F1.",
                note_style,
            )
        )
        story.append(Spacer(1, 3))
        test_header = [
            "model",
            "acc",
            "macro-P",
            "macro-R",
            "macro-F1",
            "errors",
            "mean conf.",
            "below review thr.",
            "top confusion pair",
            "official",
        ]
        test_rows: list[list[str]] = []
        official_index = None
        for row in by_test:
            test = row["test"]
            if row["official"]:
                official_index = 1 + len(test_rows)
            test_rows.append(
                [
                    row["display"],
                    _fmt(test["test_accuracy"]),
                    _fmt(test["test_macro_precision"]),
                    _fmt(test["test_macro_recall"]),
                    _fmt(test["test_macro_f1"]),
                    f"{test['test_errors']}/{test['test_images']}",
                    _fmt(test["mean_confidence"]),
                    str(test["below_review_threshold_images"]),
                    str(test.get("top_confusion_pair", "-")),
                    "yes" if row["official"] else "no",
                ]
            )
        story.append(
            _table(
                test_header,
                test_rows,
                [50 * mm, 17 * mm, 19 * mm, 19 * mm, 21 * mm, 17 * mm, 22 * mm, 25 * mm, 40 * mm, 17 * mm],
                right_cols=(1, 2, 3, 4, 5, 7),
                center_cols=(9,),
                highlight_row=official_index,
            )
        )
        story.append(Spacer(1, 5 * mm))
        if test_per_class:
            heatmap = _chart_test_per_class_heatmap(
                test_per_class,
                {row["name"]: row["test_f1"] for row in by_test},
                class_names,
                figures / "test_per_class.png",
            )
            story.append(_image(heatmap, page_width, 60 * mm))
    else:
        story.append(
            Paragraph(
                "The test diagnostic has not been generated yet. Run "
                "<font face='Courier'>python scripts/evaluate_all_models_on_test.py</font> "
                "and regenerate this report.",
                body_style,
            )
        )
    story.append(PageBreak())

    # --- production detail ---------------------------------------------------
    story.append(
        Paragraph(f"4. Production model in detail: {official['display'] if official else official_name}", heading_style)
    )
    official_entry = results[official_name]
    val_metrics = official_entry["validation_metrics"]

    per_class_header = [
        "class",
        "val support",
        "val P",
        "val R",
        "val F1",
        "test support",
        "test P",
        "test R",
        "test F1",
    ]
    per_class_rows: list[list[str]] = []
    test_metrics = (frozen or {}).get("test_metrics", {})
    for name in class_names:
        val_cell = val_metrics["per_class"][name]
        test_cell = test_metrics.get("per_class", {}).get(name, {})
        per_class_rows.append(
            [
                name,
                str(val_cell["support"]),
                _fmt(val_cell["precision"]),
                _fmt(val_cell["recall"]),
                _fmt(val_cell["f1"]),
                str(test_cell.get("support", "-")),
                _fmt(test_cell.get("precision")),
                _fmt(test_cell.get("recall")),
                _fmt(test_cell.get("f1")),
            ]
        )
    story.append(
        _table(
            per_class_header,
            per_class_rows,
            [30 * mm, 24 * mm, 22 * mm, 22 * mm, 22 * mm, 26 * mm, 22 * mm, 22 * mm, 22 * mm],
            right_cols=(1, 2, 3, 4, 5, 6, 7, 8),
            font_size=8,
        )
    )
    story.append(Spacer(1, 5 * mm))

    # calibration / uncertainty
    story.append(Paragraph("Calibration and uncertainty", heading_style))
    cal_header = ["split / state", "NLL", "ECE (10 bins)", "Brier", "mean confidence", "accuracy"]
    cal_rows: list[list[str]] = []
    if calibration:
        for state, label in (("uncalibrated", "validation, raw"), ("calibrated", "validation, temperature-scaled")):
            block = calibration.get(state, {})
            cal_rows.append(
                [
                    label,
                    _fmt(block.get("nll")),
                    _fmt(block.get("ece_10_bins")),
                    _fmt(block.get("brier_multiclass")),
                    _fmt(block.get("mean_confidence")),
                    _fmt(block.get("accuracy")),
                ]
            )
        cal_rows.insert(
            1,
            [
                f"temperature = {calibration.get('temperature', '-'):.3f}",
                "-",
                "-",
                "-",
                "-",
                "-",
            ],
        )
    if frozen:
        block = frozen.get("test_uncertainty", {})
        cal_rows.append(
            [
                "test, temperature-scaled",
                _fmt(block.get("nll")),
                _fmt(block.get("ece_10_bins")),
                _fmt(block.get("brier_multiclass")),
                _fmt(block.get("mean_confidence")),
                _fmt(block.get("accuracy")),
            ]
        )
    if cal_rows:
        story.append(
            _table(
                cal_header,
                cal_rows,
                [58 * mm, 24 * mm, 30 * mm, 24 * mm, 34 * mm, 24 * mm],
                right_cols=(1, 2, 3, 4, 5),
                font_size=8,
            )
        )
    else:
        story.append(Paragraph("Calibration artifacts not found.", note_style))

    # selective prediction
    if frozen and frozen.get("test_selective"):
        selective = frozen["test_selective"]
        threshold = frozen.get("review_threshold", {})
        selective_rows = [
            ["review threshold (selected on validation)", _fmt(threshold.get("threshold"))],
            ["accepted / review", f"{selective['accepted_count']} / {selective['review_count']}"],
            ["automatic coverage", _fmt(selective["automatic_coverage"])],
            ["review rate", _fmt(selective["review_rate"])],
            ["automatic accuracy", _fmt(selective["automatic_accuracy"])],
            ["review accuracy", _fmt(selective["review_accuracy"])],
            [
                "errors flagged for review",
                f"{selective['overall_correctly_flagged']} / {selective['total_errors']}",
            ],
        ]
        story.append(
            KeepTogether(
                [
                    Spacer(1, 4 * mm),
                    Paragraph("Selective prediction on the test split", heading_style),
                    _table(
                        ["field", "value"],
                        selective_rows,
                        [70 * mm, 60 * mm],
                        right_cols=(1,),
                        font_size=8,
                    ),
                ]
            )
        )

    story.append(PageBreak())

    # --- confusion matrices --------------------------------------------------
    story.append(
        Paragraph("5. Confusion matrices - production model", heading_style)
    )
    panels: list[tuple[str, dict[str, Any]]] = [
        (f"validation ({sizes['validation']} images)", val_metrics)
    ]
    if test_metrics:
        panels.append((f"test ({sizes['test']} images)", test_metrics))
    confusion = _draw_confusion_panels(panels, class_names, figures / "confusion.png")
    story.append(_image(confusion, page_width, 98 * mm))
    story.append(Spacer(1, 4 * mm))
    if frozen and frozen.get("mutual_confusion_pairs"):
        top_pairs = frozen["mutual_confusion_pairs"][:5]
        pair_rows = [
            [f"{pair['class_a']} <-> {pair['class_b']}", str(pair["a_as_b"]), str(pair["b_as_a"]), _fmt(pair["pair_confusion"])]
            for pair in top_pairs
        ]
        story.append(
            KeepTogether(
                [
                    Paragraph("Most frequent mutual confusions (test split)", heading_style),
                    _table(
                        ["class pair", "A as B", "B as A", "pair confusion"],
                        pair_rows,
                        [60 * mm, 25 * mm, 25 * mm, 30 * mm],
                        right_cols=(1, 2, 3),
                        font_size=8,
                    ),
                ]
            )
        )

    story.append(PageBreak())

    # --- methodology ---------------------------------------------------------
    story.append(Paragraph("6. Methodology and caveats", heading_style))
    integrity = (test_summary or {}).get("integrity_check")
    caveats = [
        (
            "Model selection used the validation split only (stratified "
            f"{config.get('validation_fraction', 0.2):.0%} of the training images, seed "
            f"{config.get('seed', 42)}); the best epoch of each run is chosen by validation "
            "macro-F1."
        ),
        (
            "The test split is evaluated after selection, once per model, and is reported "
            "here as a diagnostic. It must not be used to pick a model or tune hyperparameters."
        ),
        (
            "All models share the same class mapping, stratified split and evaluation "
            "transform; differences between rows are attributable to the training "
            "configuration named by each experiment."
        ),
        (
            "ECE uses 10 equal-width confidence bins; production-model NLL/Brier are "
            "computed after temperature scaling, with the temperature fitted on the "
            "validation split."
        ),
        (
            "Integrity check: recomputed test macro-F1 matches the locked value "
            f"(abs difference {integrity['abs_difference']})."
            if integrity and integrity.get("passed")
            else "Test-integrity artifact not found."
        ),
        (
            "Per-image test predictions for non-production models are diagnostic outputs "
            "of scripts/evaluate_all_models_on_test.py; their per-class F1 values are "
            "recomputed from those predictions."
        ),
    ]
    for item in caveats:
        story.append(Paragraph(item, bullet_style, bulletText="\u2022"))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "Generated by scripts/generate_pdf_report.py "
            f"on {_now().strftime('%Y-%m-%d %H:%M:%S')}.",
            note_style,
        )
    )

    document = SimpleDocTemplate(
        str(output),
        pagesize=PAGE,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=14 * mm,
        title="Traffic Vehicle Classification - model evaluation report",
        author="traffic_classifier",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    readme_path = (
        ROOT / config.get("artifacts_dir", "artifacts") / "results" / "all_models_test" / "README.md"
    )
    if _update_readme(
        readme_path,
        pdf_path=output,
        model_count=len(joined),
        official_name=official_name,
        sizes=sizes,
        fingerprint=(production or {}).get("run_fingerprint") or metadata_fingerprint(results) or "-",
    ):
        print(f"readme     : {readme_path.as_posix()} (PDF section refreshed)")
    else:
        print(
            "readme     : skipped "
            f"({readme_path.parent.as_posix()} does not exist; "
            "run `python scripts/evaluate_all_models_on_test.py` first)"
        )
    print(f"report     : {output.as_posix()}")
    print(f"models     : {len(joined)} ({'validation + test' if by_test else 'validation only'})")
    print(f"production : {official_name}")


def metadata_fingerprint(results: dict[str, Any]) -> str | None:
    for entry in results.values():
        fingerprint = (entry.get("metadata") or {}).get("run_fingerprint")
        if fingerprint:
            return fingerprint
    return None


def _update_readme(
    readme_path: Path,
    *,
    pdf_path: Path,
    model_count: int,
    official_name: str,
    sizes: dict[str, Any],
    fingerprint: str,
) -> bool:
    """Refresh the `## PDF report` section of `all_models_test/README.md`.

    The section is replaced in place on every run so the README always points
    at the current PDF with its hash; the rest of the README is left alone.
    Returns False when the directory does not exist yet.
    """

    if not readme_path.parent.is_dir():
        return False
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    try:
        relative = pdf_path.relative_to(ROOT).as_posix()
    except ValueError:
        relative = pdf_path.as_posix()
    split = f"{sizes['train']} train / {sizes['validation']} validation"
    if sizes.get("test"):
        split += f" / {sizes['test']} test"
    section = (
        "## PDF report\n"
        "\n"
        f"`{relative}` - full evaluation report covering validation and test metrics\n"
        f"for all {model_count} models, generated after the test diagnostic:\n"
        "\n"
        "    python scripts/generate_pdf_report.py\n"
        "\n"
        "| Field | Value |\n"
        "| --- | --- |\n"
        f"| Generated | {_now().strftime('%Y-%m-%d %H:%M:%S')} |\n"
        f"| SHA-256 | `{digest}` |\n"
        f"| Models covered | {model_count} |\n"
        f"| Production model | `{official_name}` |\n"
        f"| Dataset split | {split} |\n"
        f"| Run fingerprint | `{fingerprint}` |\n"
    )
    if readme_path.is_file():
        text = readme_path.read_text(encoding="utf-8")
        start = text.find("## PDF report")
        if start == -1:
            updated = text.rstrip("\n") + "\n\n" + section
        else:
            following = text.find("\n## ", start + 1)
            end = following + 1 if following != -1 else len(text)
            updated = text[:start] + section + "\n" + text[end:]
    else:
        updated = "# All-model test diagnostic\n\n" + section
    readme_path.write_text(updated, encoding="utf-8")
    return True


def _findings(
    *,
    joined: list[dict[str, Any]],
    by_val: list[dict[str, Any]],
    by_test: list[dict[str, Any]],
    official: dict[str, Any] | None,
    frozen: dict[str, Any] | None,
    test_summary: dict[str, Any] | None,
) -> list[str]:
    findings: list[str] = []
    best_val = by_val[0]
    findings.append(
        f"Highest validation macro-F1: <b>{best_val['display']}</b> "
        f"({best_val['val_f1']:.4f} at epoch {best_val['best_epoch']})."
    )
    if official:
        findings.append(
            f"Selected production model: <b>{official['display']}</b> "
            f"(validation macro-F1 {official['val_f1']:.4f}"
            + (
                f", test macro-F1 {official['test_f1']:.4f} "
                f"on {official['test']['test_images']} images, "
                f"{official['test']['test_errors']} errors)"
                if official.get("test")
                else ")"
            )
            + "."
        )
    if by_test:
        best_test = by_test[0]
        ties = [row for row in by_test if row["test_f1"] == best_test["test_f1"]]
        names = ", ".join(row["display"] for row in ties)
        findings.append(
            f"Highest test macro-F1 (diagnostic): {names} "
            f"({best_test['test_f1']:.4f})."
        )
        spread = by_val[0]["val_f1"] - by_val[-1]["val_f1"]
        findings.append(
            f"Validation macro-F1 spans {by_val[-1]['val_f1']:.4f} to "
            f"{by_val[0]['val_f1']:.4f} across {len(by_val)} models (spread "
            f"{spread:.4f})."
        )
    if frozen and frozen.get("test_metrics"):
        metrics = frozen["test_metrics"]
        findings.append(
            f"Production test performance: accuracy {metrics['accuracy']:.4f}, "
            f"macro-F1 {metrics['macro_f1']:.4f}, "
            f"{frozen.get('error_count', '-')} errors."
        )
    if test_summary and test_summary.get("official_evaluation"):
        protocol = test_summary["official_evaluation"]
        findings.append(
            f"Selection protocol: {protocol.get('selected_using', '-')} "
            f"-> {protocol.get('selected_experiment', '-')}; "
            f"{protocol.get('test_evaluations', '-')} model(s) evaluated on the test split."
        )
    return findings


if __name__ == "__main__":
    main()
