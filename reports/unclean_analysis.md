# Cleaned `unclean` and Unseen-Class Analysis

`unclean` is never used for optimization or validation. A second content-aware pass removes every pixel-identical copy of train or test, producing **724** retained images from 724; **0** exclusions are recorded with their source. This includes the `train/vanet` versus `unclean/neysan` label conflict, which is excluded rather than silently relabelled.

The retained set has **724** known-class examples and **0** `neysan` examples. On the known subset, accuracy is **0.6450** and macro-F1 is **0.6342**. This is diagnostic data, not another model-selection test.

No `neysan` samples remain in the `unclean` split after cleaning (its folder was emptied by the duplicate-removal pass), so the unseen-class branch is skipped. `neysan` is still conceptually treated as an unseen/human-review class, never added as a ninth training label. Standard softmax confidence is not an out-of-distribution detector, so any future unseen-class analysis should still route low-confidence cases to human review.
