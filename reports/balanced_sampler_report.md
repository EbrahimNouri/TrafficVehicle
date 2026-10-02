# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `142`, 14 examples are retained from each selected minority class (`kamyun`, `minibus`, `savari`, `vanet`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **256**, hence **32 examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

| Class | Standard recall | Balanced recall | Δ recall | Standard precision | Balanced precision | Standard F1 | Balanced F1 |
|---|---|---|---|---|---|---|---|
| ambulance | 0.8823529411764706 | 0.706 | -0.176 | 0.500 | 0.750 | 0.638 | 0.727 |
| autobus | 0.85 | 0.650 | -0.200 | 0.531 | 0.448 | 0.654 | 0.531 |
| kamyun | 0.0 | 0.421 | +0.421 | 0.000 | 0.667 | 0.000 | 0.516 |
| kamyunet | 0.6666666666666666 | 0.571 | -0.095 | 0.452 | 0.571 | 0.538 | 0.571 |
| minibus | 0.0 | 0.632 | +0.632 | 0.000 | 0.545 | 0.000 | 0.585 |
| savari | 0.5714285714285714 | 0.619 | +0.048 | 0.500 | 0.591 | 0.533 | 0.605 |
| taxi | 0.8421052631578947 | 0.842 | +0.000 | 0.762 | 0.615 | 0.800 | 0.711 |
| vanet | 0.7 | 0.500 | -0.200 | 0.778 | 0.833 | 0.737 | 0.625 |

- Standard: accuracy **0.5723**, macro-F1 **0.4876**.
- Balanced: accuracy **0.6084**, macro-F1 **0.6089**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
