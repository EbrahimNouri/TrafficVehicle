# Traffic Vehicle Classification with CNNs and Transfer Learning

A complete, reproducible implementation of `Project-Definition2-traffic-vehicle.ipynb` for eight traffic-camera vehicle classes: `ambulance`, `autobus`, `kamyun`, `kamyunet`, `minibus`, `savari`, `taxi`, and `vanet`.

The repository emphasizes **leakage prevention, controlled experiments, honest frozen-test evaluation, uncertainty-aware JSON inference, and reproducible artifacts** rather than a single headline accuracy.

## Reproduce from scratch

Python 3.11+ is supported (Python 3.12 is used for the recorded run).

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name traffic-vehicle-python312 --display-name "Python 3.12 (Traffic Vehicle)"
.\.venv\Scripts\python -m traffic_classifier.pipeline --config configs/default.json
```

`main.py` is an equivalent PyCharm entry point. Compatible completed experiment checkpoints are reused automatically. Every epoch also writes an atomic rolling `<experiment>.last.pt` checkpoint containing model, optimizer, scheduler, best weights, history, elapsed time, and RNG/DataLoader state; if the process is interrupted, rerun the same command to continue at the next epoch. Pass `--force` only when intentionally discarding existing runs and starting fresh (a later normal command can still resume an interrupted forced run).

The supplied `dataset/` directory is ignored by Git. It must contain `train/`, `test/`, and `unclean/` exactly as described in the assignment.

## Make a prediction

After training:

```powershell
python predict.py path/to/vehicle.jpg
```

Example output shape:

```json
{
  "predicted_class": "ambulance",
  "confidence": 0.91,
  "probabilities": {
    "ambulance": 0.91,
    "autobus": 0.01,
    "kamyun": 0.02,
    "kamyunet": 0.01,
    "minibus": 0.01,
    "savari": 0.01,
    "taxi": 0.02,
    "vanet": 0.01
  },
  "needs_review": false
}
```

`predict_image()` in `traffic_classifier/predict.py` is the equivalent callable API. Production probabilities are temperature-scaled **Cross-Entropy softmax** scores. The review threshold is selected on validation data, never test data.

## Protocol

1. **Audit before splitting.** Each split is loaded independently. Unreadable files, unsupported formats, small images, aspect ratios, formats, dimensions, class counts, and representative examples are recorded.
2. **Use content hashes.** The project computes encoded-file SHA-256 and an encoding-independent hash of decoded, EXIF-oriented RGB pixels. Both identify the same 24 duplicate groups in this supplied copy; the canonical pixel result is the authoritative check.
3. **Freeze test.** No original train/test image is removed. The test set is excluded from every decision and evaluated once after selecting on validation macro-F1. Output/error archives and the production checkpoint are SHA-256-bound to the run manifest.
4. **Create a fixed split.** Seed 42 creates a per-class 80/20 training/validation partition; exact indices and paths are saved.
5. **Compare fairly.** Single-factor ablations use the same seed, split, initialization, optimizer, learning rate, and epoch budget. Only the named factor changes.
6. **Simulate imbalance reproducibly.** Four classes retain 14 examples while the others retain 40. Exact indices are recorded. A custom sampler puts four examples from every class into each size-32 batch.
7. **Treat BCE correctly.** BCE gets one-hot `float32` targets; sigmoid is applied only after training; multiclass predictions use `argmax(logits)`.
8. **Transfer conservatively.** Standard pretrained ResNet18 is compared as frozen feature extraction and head-first/layer4 fine-tuning with separate learning-rate groups.
9. **Quantify uncertainty.** Validation selects scalar temperature and a threshold that targets 80% automatic coverage. `unclean/neysan` remains unseen and is analyzed separately, never added as a ninth training label.
10. **Prevent concurrent publication.** A per-artifacts-directory OS lock permits one pipeline writer. The default prediction API serves only a hash-verified `final_model.pt` with a matching run manifest, source checkpoint, transform metadata, and test-evidence hashes.

## Experiments

| Experiment | Controlled question |
|---|---|
| CNN baseline | Reproducible four-block reference |
| No augmentation | Does realistic train-time variation help? |
| Dropout 0.3 / 0.5 | Does dropout reduce overfitting? |
| Max vs. average pooling | Which spatial summary helps? |
| Weight decay 0 / `1e-4` | Does AdamW regularization help? |
| Fixed LR vs. StepLR | Does scheduling help? |
| Standard vs. balanced batches | How does class-balanced sampling affect minority recall? |
| Cross-entropy vs. BCE | How do task assumptions affect a single-label problem? |
| ResNet feature extraction | Is ImageNet representation sufficient? |
| ResNet fine-tuning | Does adapting `layer4` help? |
| Best regularized combination | Can validated regularization choices combine effectively? |

The best Cross-Entropy model is selected **only by validation macro-F1**. BCE is an educational comparison and cannot become the production model.

## Important generated artifacts

### Reports

- `reports/final_report.md` — executive summary, required comparison table, final test, and completed checklist
- `reports/data_audit.md` — counts, dimensions, quality findings, and every exclusion
- `reports/ablation_report.md` — controlled regularization/scheduler comparison
- `reports/balanced_sampler_report.md` — simulated imbalance and per-class effects
- `reports/loss_comparison.md` — CE/BCE assumptions, metrics, and confidence
- `reports/transfer_learning_report.md` — parameters, stages, and LRs
- `reports/error_analysis.md` — frozen-test metrics, pair ranking, threshold, and merge decision
- `reports/unclean_analysis.md` — duplicate removal and unseen `neysan`

### Machine-readable evidence

- `artifacts/run_manifest.json` — resolved configuration, split, data, source, runtime, device, and pretrained-weight fingerprint
- `artifacts/production_ready.json` — publication hashes for the final model and frozen-test evidence
- `artifacts/audit/summary.json`
- `artifacts/audit/canonical_duplicate_groups.json`
- `artifacts/audit/cleaned_unclean_exclusions.json`
- `artifacts/splits/train_validation.json`
- `artifacts/splits/simulated_imbalance.json`
- `artifacts/results/experiment_results.json`
- `artifacts/results/experiment_summary.csv`
- `artifacts/results/balanced_batch_comparison.json`
- `artifacts/results/validation_uncertainty.json`
- `artifacts/results/frozen_test_evaluation.json`
- `artifacts/results/test_errors.csv`
- `artifacts/results/unclean_analysis.json`
- `artifacts/results/unclean_predictions.csv`
- `artifacts/checkpoints/<experiment>.pt` — best validation checkpoint, rewritten whenever a new best epoch is reached
- `artifacts/checkpoints/<experiment>.last.pt` — atomic per-epoch rolling checkpoint used for exact resume
- `artifacts/checkpoints/final_model.pt` — selected production checkpoint (ignored by Git because it is large)

### Figures

- `reports/figures/representative_images.png` — 16 labeled examples
- `reports/figures/image_size_distributions.png`
- `reports/figures/training_curves.png`
- `reports/figures/per_class_comparison.png`
- `reports/figures/final_confusion_matrices.png`
- `reports/figures/validation_risk_coverage.png`
- `reports/figures/misclassified_test_images.png` — 12 labeled errors
- `reports/figures/unclean_low_confidence_examples.png`

## Notebook

`Project-Definition2-traffic-vehicle.ipynb` combines the completed analysis, executable workflow, measured tables, visualizations, prediction example, and final rubric checklist. The assignment text is retained as an appendix. To execute it:

```powershell
jupyter nbconvert --to notebook --execute --inplace Project-Definition2-traffic-vehicle.ipynb
```

The notebook reuses completed experiment artifacts. For a clean retraining run, execute the pipeline command first or remove the ignored checkpoint directory intentionally.

## Tests

```powershell
pytest
```

Tests cover exact balanced-batch composition, deterministic sampling, pixel-content hashing, BCE target encoding, CNN output/parameter accounting, distinct ablation factors, deterministic evaluation transforms, path alignment, and checkpoint resume compatibility. A separate interrupted-process smoke test verified epoch-level state restoration.

## Repository structure

```text
configs/default.json                 Reproducible configuration
traffic_classifier/
  audit.py                           Dataset quality and duplicate audit
  data.py                            ImageFolder, transforms, split, sampler
  models.py                          Four-block CNN and ResNet18 builders
  engine.py                          Training/validation loops and checkpoints
  metrics.py                         Metrics, calibration, uncertainty, pair ranking
  plotting.py                        Reproducible report figures
  pipeline.py                        End-to-end controlled experiment workflow
  predict.py                         Review-aware JSON inference
  reporting.py                       Markdown/HTML reports
predict.py                           CLI convenience wrapper
main.py                              PyCharm/CLI convenience wrapper
tests/test_core.py                   Unit tests
```

## Limitations

- The clean training set has only 400 images, so validation estimates have meaningful uncertainty; differences of a few points are not necessarily decisive.
- Cropped vehicles with severe occlusion, blur, unusual pose, or very small apparent size can be intrinsically ambiguous.
- Softmax confidence is not a guaranteed out-of-distribution detector. `needs_review` is a risk-control mechanism, not proof that an accepted prediction is correct.
- ImageNet pretraining uses standard ResNet18 preprocessing at 224×224 while retaining `conv1` and `maxpool`.
