# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `142`, 14 examples are retained from each selected minority class (`kamyun`, `minibus`, `savari`, `vanet`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **32**, hence **4 examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

| Class | Standard recall | Balanced recall | Δ recall | Standard precision | Balanced precision | Standard F1 | Balanced F1 |
|---|---|---|---|---|---|---|---|
| ambulance | 0.9 | 0.900 | +0.000 | 0.562 | 0.750 | 0.692 | 0.818 |
| autobus | 0.9 | 1.000 | +0.100 | 0.692 | 0.714 | 0.783 | 0.833 |
| kamyun | 0.6 | 0.400 | -0.200 | 0.667 | 1.000 | 0.632 | 0.571 |
| kamyunet | 0.3 | 0.600 | +0.300 | 0.500 | 0.545 | 0.375 | 0.571 |
| minibus | 0.1 | 0.400 | +0.300 | 1.000 | 0.571 | 0.182 | 0.471 |
| savari | 0.5 | 0.700 | +0.200 | 0.625 | 0.778 | 0.556 | 0.737 |
| taxi | 0.9 | 1.000 | +0.100 | 0.692 | 1.000 | 0.783 | 1.000 |
| vanet | 0.9 | 1.000 | +0.100 | 0.643 | 0.769 | 0.750 | 0.870 |

- Standard: accuracy **0.6375**, macro-F1 **0.5939**.
- Balanced: accuracy **0.7500**, macro-F1 **0.7339**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
