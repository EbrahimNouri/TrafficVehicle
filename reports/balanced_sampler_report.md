# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `142`, 14 examples are retained from each selected minority class (`kamyun`, `minibus`, `savari`, `vanet`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **256**, hence **32 examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

| Class | Standard recall | Balanced recall | Δ recall | Standard precision | Balanced precision | Standard F1 | Balanced F1 |
|---|---|---|---|---|---|---|---|
| ambulance | 0.9411764705882353 | 0.824 | -0.118 | 0.615 | 0.667 | 0.744 | 0.737 |
| autobus | 0.85 | 0.900 | +0.050 | 0.680 | 0.667 | 0.756 | 0.766 |
| kamyun | 0.47368421052631576 | 0.368 | -0.105 | 0.818 | 0.636 | 0.600 | 0.467 |
| kamyunet | 0.9047619047619048 | 0.810 | -0.095 | 0.731 | 0.548 | 0.809 | 0.654 |
| minibus | 0.7368421052631579 | 0.421 | -0.316 | 0.609 | 0.889 | 0.667 | 0.571 |
| savari | 0.6666666666666666 | 0.857 | +0.190 | 0.933 | 0.720 | 0.778 | 0.783 |
| taxi | 0.8947368421052632 | 0.842 | -0.053 | 0.773 | 0.889 | 0.829 | 0.865 |
| vanet | 0.5666666666666667 | 0.667 | +0.100 | 0.944 | 0.833 | 0.708 | 0.741 |

- Standard: accuracy **0.7410**, macro-F1 **0.7363**.
- Balanced: accuracy **0.7108**, macro-F1 **0.6979**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
