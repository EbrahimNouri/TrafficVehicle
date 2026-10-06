# Data Audit and Leakage Prevention

## Protocol

- Audit completed before validation creation and model fitting.
- Each split is loaded separately; `unclean` is never merged into training.
- The audit computes both encoded-file SHA-256 and a canonical decoded, EXIF-oriented RGB pixel hash. The latter is encoding-independent by design. In this supplied copy both methods identify the same 0 duplicate groups, but the canonical result remains the authoritative leakage check.
- Files smaller than 64 pixels on either side, unreadable files, unsupported formats, and extreme aspect ratios would be flagged.
- Frozen train/test exclusions: **8** (Keep the frozen-test copy; quarantine the pixel-identical train copy to `dataset/quarantine/split_duplicates/`.). Current train: **1162**; original test: **64**.

## Original splits

| Split | Images | Class counts | Width range | Height range | Unique pixel hashes |
|---|---|---|---|---|---|
| train | 1162 | ambulance=127, autobus=140, kamyun=137, kamyunet=145, minibus=138, savari=148, taxi=137, vanet=190 | 120–545 | 156–938 | 1162 |
| test | 64 | ambulance=8, autobus=8, kamyun=8, kamyunet=8, minibus=8, savari=8, taxi=8, vanet=8 | 150–508 | 192–450 | 64 |
| unclean | 724 | ambulance=52, autobus=42, kamyun=145, kamyunet=166, minibus=68, savari=94, taxi=48, vanet=109 | 123–548 | 166–956 | 724 |

Quality findings:

| Finding | Count |
|---|---|
| none | 0 |

All 1,950 source files decode successfully, use JPEG format, and have the same eight classes except for `unclean/neysan`. Dimensions vary substantially (as expected for traffic-camera crops), but no image is unusually small under the declared rule. Statistical 1.5×IQR width/height review counts are recorded per split in `summary.json`; these are review flags, not automatic exclusions.

## Canonical-pixel duplicate findings

- Encoded-file SHA-256 unique values: **1,950**; duplicate groups: **0**.
- Canonical RGB pixel unique values: **1,950**; duplicate groups: **0**; extra copies: **0**.
- Cross-split groups: **0**; label conflicts: **0**; train–test overlap groups: **0** (the workflow fails closed if nonzero).

| Group | Members | Conflict? |
|---|---|---|

Filename equality alone was not used. For example, `214844236.jpg` is a pixel-identical `train/vanet` and `unclean/neysan` image, demonstrating the apparent label conflict. It is excluded from the cleaned `unclean` analysis rather than relabelled or used for training.

## Cleaning decisions

| Dataset | Excluded | Clean retained |
|---|---|---|
| train | 8 | 1162 |
| test (frozen) | 0 | 64 |
| unclean | 0 | 724 |

Retained `unclean` class counts: | Class | Count |
|---|---|
| ambulance | 52 |
| autobus | 42 |
| kamyun | 145 |
| kamyunet | 166 |
| minibus | 68 |
| savari | 94 |
| taxi | 48 |
| vanet | 109 |.

Every exclusion is recorded in `artifacts/audit/cleaned_unclean_exclusions.json`. All exclusions are duplicate copies inside `unclean`; none belongs to the original train or test split. This avoids leakage without silently changing the frozen evaluation sample.

## Representative examples

The reproducible 16-image gallery (two per known class) is saved as `reports/figures/representative_images.png` from these exact paths:

- `dataset/train/ambulance/198727855.jpg`
- `dataset/train/ambulance/199038575.jpg`
- `dataset/train/autobus/193229239.jpg`
- `dataset/train/autobus/193813673.jpg`
- `dataset/train/kamyun/196530974.jpg`
- `dataset/train/kamyun/198661910.jpg`
- `dataset/train/kamyunet/193903806.jpg`
- `dataset/train/kamyunet/194270924.jpg`
- `dataset/train/minibus/194328969.jpg`
- `dataset/train/minibus/195315678.jpg`
- `dataset/train/savari/193865576.jpg`
- `dataset/train/savari/195800847.jpg`
- `dataset/train/taxi/196935725.jpg`
- `dataset/train/taxi/197011132.jpg`
- `dataset/train/vanet/194285934.jpg`
- `dataset/train/vanet/194840733.jpg`
