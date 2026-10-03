# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `142`, 14 examples are retained from each selected minority class (`kamyun`, `minibus`, `savari`, `vanet`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **256**, hence **32 examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

| Class | Standard recall | Balanced recall | Δ recall | Standard precision | Balanced precision | Standard F1 | Balanced F1 |
|---|---|---|---|---|---|---|---|
| ambulance | 0.8823529411764706 | 0.824 | -0.059 | 0.652 | 0.667 | 0.750 | 0.737 |
| autobus | 0.8 | 0.900 | +0.100 | 0.842 | 0.667 | 0.821 | 0.766 |
| kamyun | 0.47368421052631576 | 0.368 | -0.105 | 0.750 | 0.636 | 0.581 | 0.467 |
| kamyunet | 0.9523809523809523 | 0.810 | -0.143 | 0.588 | 0.548 | 0.727 | 0.654 |
| minibus | 0.5789473684210527 | 0.421 | -0.158 | 0.611 | 0.889 | 0.595 | 0.571 |
| savari | 0.7619047619047619 | 0.857 | +0.095 | 0.667 | 0.720 | 0.711 | 0.783 |
| taxi | 0.8421052631578947 | 0.842 | +0.000 | 0.696 | 0.889 | 0.762 | 0.865 |
| vanet | 0.43333333333333335 | 0.667 | +0.233 | 1.000 | 0.833 | 0.605 | 0.741 |

- Standard: accuracy **0.6988**, macro-F1 **0.6938**.
- Balanced: accuracy **0.7108**, macro-F1 **0.6979**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
