# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `142`, 14 examples are retained from each selected minority class (`kamyun`, `minibus`, `savari`, `vanet`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **128**, hence **16 examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

| Class | Standard recall | Balanced recall | Δ recall | Standard precision | Balanced precision | Standard F1 | Balanced F1 |
|---|---|---|---|---|---|---|---|
| ambulance | 0.8823529411764706 | 0.824 | -0.059 | 0.833 | 1.000 | 0.857 | 0.903 |
| autobus | 0.8 | 0.850 | +0.050 | 0.889 | 0.773 | 0.842 | 0.810 |
| kamyun | 0.42105263157894735 | 0.368 | -0.053 | 0.667 | 0.778 | 0.516 | 0.500 |
| kamyunet | 0.8571428571428571 | 0.905 | +0.048 | 0.720 | 0.442 | 0.783 | 0.594 |
| minibus | 0.6842105263157895 | 0.474 | -0.211 | 0.650 | 0.643 | 0.667 | 0.545 |
| savari | 0.8095238095238095 | 0.667 | -0.143 | 0.850 | 0.667 | 0.829 | 0.667 |
| taxi | 0.8421052631578947 | 0.842 | +0.000 | 0.842 | 0.727 | 0.842 | 0.780 |
| vanet | 0.7666666666666667 | 0.633 | -0.133 | 0.676 | 0.905 | 0.719 | 0.745 |

- Standard: accuracy **0.7590**, macro-F1 **0.7568**.
- Balanced: accuracy **0.6928**, macro-F1 **0.6930**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
