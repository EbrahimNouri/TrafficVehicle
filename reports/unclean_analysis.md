# Cleaned `unclean` and Unseen-Class Analysis

`unclean` is never used for optimization or validation. A second content-aware pass removes every pixel-identical copy of train or test, producing **159** retained images from 183; **24** exclusions are recorded with their source. This includes the `train/vanet` versus `unclean/neysan` label conflict, which is excluded rather than silently relabelled.

The retained set has **109** known-class examples and **50** `neysan` examples. On the known subset, accuracy is **0.6789** and macro-F1 is **0.6719**. This is diagnostic data, not another model-selection test.

For unseen `neysan`, mean top-class confidence is **0.9086** and **7 / 50 (14.0%)** fall below the validation review threshold. Predicted known-class counts are:

| Prediction | Count |
|---|---|
| ambulance | 0 |
| autobus | 3 |
| kamyun | 1 |
| kamyunet | 0 |
| minibus | 0 |
| savari | 0 |
| taxi | 1 |
| vanet | 45 |

`neysan` is analyzed as unseen/human-review data, never added as a ninth class. Standard softmax confidence is not an out-of-distribution detector: an unseen image can receive a high known-class score. The observed confidence pattern is therefore a risk signal, not proof of semantic correctness. `figures/unclean_low_confidence_examples.png` shows the 12 least-confident `neysan` cases and `artifacts/results/unclean_predictions.csv` contains every prediction.
