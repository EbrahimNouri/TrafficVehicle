# Cleaned `unclean` and Unseen-Class Analysis

`unclean` is never used for optimization or validation. This run found no image files in the `unclean` split, so the duplicate-removal pass and the unseen-class analysis were **skipped**. No numbers are reported here: a skipped step is not a measured result.

Reason: the unclean split at dataset/unclean holds no image files, so the cleaned-unclean and unseen-class analysis was skipped. `artifacts/results/unclean_analysis.json` records `skipped: true` and `artifacts/results/unclean_predictions.csv` is written with headers only. Add images to the `unclean` split and re-run to populate this section.
