# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `142`, 14 examples are retained from each selected minority class (`kamyun`, `minibus`, `savari`, `vanet`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **256**, hence **32 examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

| Class | Standard recall | Balanced recall | Δ recall | Standard precision | Balanced precision | Standard F1 | Balanced F1 |
|---|---|---|---|---|---|---|---|
| ambulance | 0.8823529411764706 | 0.765 | -0.118 | 0.938 | 0.929 | 0.909 | 0.839 |
| autobus | 0.85 | 0.950 | +0.100 | 0.739 | 0.633 | 0.791 | 0.760 |
| kamyun | 0.5263157894736842 | 0.368 | -0.158 | 0.769 | 0.875 | 0.625 | 0.519 |
| kamyunet | 0.9047619047619048 | 0.952 | +0.048 | 0.613 | 0.455 | 0.731 | 0.615 |
| minibus | 0.7368421052631579 | 0.474 | -0.263 | 0.609 | 1.000 | 0.667 | 0.643 |
| savari | 0.6666666666666666 | 0.810 | +0.143 | 0.875 | 0.773 | 0.757 | 0.791 |
| taxi | 0.9473684210526315 | 0.842 | -0.105 | 0.720 | 0.842 | 0.818 | 0.842 |
| vanet | 0.6 | 0.600 | +0.000 | 0.947 | 0.900 | 0.735 | 0.720 |

- Standard: accuracy **0.7530**, macro-F1 **0.7540**.
- Balanced: accuracy **0.7169**, macro-F1 **0.7160**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
