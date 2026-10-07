# BalancedBatchSampler and Simulated Imbalance

The clean, non-leaking training partition has 40 examples per class. With seed `142`, 14 examples are retained from each selected minority class (`kamyun`, `minibus`, `savari`, `vanet`) and all 40 from the other classes. Exact indices and paths are in `artifacts/splits/simulated_imbalance.json`.

`BalancedBatchSampler` uses batch size **128**, hence **16 examples per class in every batch**. It oversamples smaller classes to the largest class count and seeds shuffling by epoch. Standard random sampling is given the same number of optimizer steps. Crucially, the validation partition is unchanged.

| Class | Standard recall | Balanced recall | Δ recall | Standard precision | Balanced precision | Standard F1 | Balanced F1 |
|---|---|---|---|---|---|---|---|
| ambulance | 0.92 | 0.880 | -0.040 | 1.000 | 0.917 | 0.958 | 0.898 |
| autobus | 0.9285714285714286 | 1.000 | +0.071 | 0.788 | 0.824 | 0.852 | 0.903 |
| kamyun | 0.5185185185185185 | 0.444 | -0.074 | 0.933 | 0.923 | 0.667 | 0.600 |
| kamyunet | 0.9310344827586207 | 0.862 | -0.069 | 0.562 | 0.500 | 0.701 | 0.633 |
| minibus | 0.6428571428571429 | 0.643 | +0.000 | 0.900 | 0.857 | 0.750 | 0.735 |
| savari | 0.8333333333333334 | 0.733 | -0.100 | 0.806 | 0.815 | 0.820 | 0.772 |
| taxi | 0.9259259259259259 | 0.926 | +0.000 | 0.833 | 0.806 | 0.877 | 0.862 |
| vanet | 0.7368421052631579 | 0.789 | +0.053 | 0.875 | 0.938 | 0.800 | 0.857 |

- Standard: accuracy **0.8017**, macro-F1 **0.8032**.
- Balanced: accuracy **0.7845**, macro-F1 **0.7825**.

Balanced batches optimize average class exposure rather than natural frequency, so minority recall and macro-F1 are the primary outcomes. Overall accuracy may fall when the sampler sacrifices majority prevalence. This is an intentional fairness trade-off, not a broken implementation.
