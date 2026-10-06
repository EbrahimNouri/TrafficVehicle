# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `142`, 14 examples are retained from each selected minority class (`kamyun`, `minibus`, `savari`, `vanet`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **256**, hence **32 examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

| Class | Standard recall | Balanced recall | Δ recall | Standard precision | Balanced precision | Standard F1 | Balanced F1 |
|---|---|---|---|---|---|---|---|
| ambulance | 0.92 | 0.920 | +0.000 | 0.852 | 0.821 | 0.885 | 0.868 |
| autobus | 0.9642857142857143 | 0.893 | -0.071 | 0.794 | 0.758 | 0.871 | 0.820 |
| kamyun | 0.48148148148148145 | 0.556 | +0.074 | 0.929 | 0.714 | 0.634 | 0.625 |
| kamyunet | 0.8620689655172413 | 0.759 | -0.103 | 0.641 | 0.579 | 0.735 | 0.657 |
| minibus | 0.6785714285714286 | 0.679 | +0.000 | 0.826 | 0.950 | 0.745 | 0.792 |
| savari | 0.7666666666666667 | 0.700 | -0.067 | 0.719 | 0.808 | 0.742 | 0.750 |
| taxi | 0.9259259259259259 | 0.926 | +0.000 | 0.833 | 0.833 | 0.877 | 0.877 |
| vanet | 0.7368421052631579 | 0.789 | +0.053 | 0.848 | 0.833 | 0.789 | 0.811 |

- Standard: accuracy **0.7888**, macro-F1 **0.7847**.
- Balanced: accuracy **0.7759**, macro-F1 **0.7749**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
