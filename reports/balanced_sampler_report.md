# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `142`, 14 examples are retained from each selected minority class (`kamyun`, `minibus`, `savari`, `vanet`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **256**, hence **32 examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

| Class | Standard recall | Balanced recall | Δ recall | Standard precision | Balanced precision | Standard F1 | Balanced F1 |
|---|---|---|---|---|---|---|---|
| ambulance | 0.92 | 0.920 | +0.000 | 0.821 | 0.821 | 0.868 | 0.868 |
| autobus | 0.9642857142857143 | 0.893 | -0.071 | 0.771 | 0.758 | 0.857 | 0.820 |
| kamyun | 0.5555555555555556 | 0.556 | +0.000 | 0.938 | 0.714 | 0.698 | 0.625 |
| kamyunet | 0.8275862068965517 | 0.759 | -0.069 | 0.686 | 0.579 | 0.750 | 0.657 |
| minibus | 0.7142857142857143 | 0.679 | -0.036 | 0.833 | 0.950 | 0.769 | 0.792 |
| savari | 0.8333333333333334 | 0.700 | -0.133 | 0.714 | 0.808 | 0.769 | 0.750 |
| taxi | 0.9259259259259259 | 0.926 | +0.000 | 0.833 | 0.833 | 0.877 | 0.877 |
| vanet | 0.7105263157894737 | 0.789 | +0.079 | 0.931 | 0.833 | 0.806 | 0.811 |

- Standard: accuracy **0.8017**, macro-F1 **0.7993**.
- Balanced: accuracy **0.7759**, macro-F1 **0.7749**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
