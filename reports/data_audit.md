# Data Audit and Leakage Prevention

## Protocol

- Audit completed before validation creation and model fitting.
- Each split is loaded separately; `unclean` is never merged into training.
- The audit computes both encoded-file SHA-256 and a canonical decoded, EXIF-oriented RGB pixel hash. The latter is encoding-independent by design. In this supplied copy both methods identify the same 23 duplicate groups, but the canonical result remains the authoritative leakage check.
- Files smaller than 64 pixels on either side, unreadable files, unsupported formats, and extreme aspect ratios would be flagged.
- Frozen train/test exclusions: **0**. Original train: **400**; original test: **400**.

## Original splits

| Split | Images | Class counts | Width range | Height range | Unique pixel hashes |
|---|---|---|---|---|---|
| train | 400 | ambulance=50, autobus=50, kamyun=50, kamyunet=50, minibus=50, savari=50, taxi=50, vanet=50 | 120–545 | 156–870 | 400 |
| test | 400 | ambulance=50, autobus=50, kamyun=50, kamyunet=50, minibus=50, savari=50, taxi=50, vanet=50 | 126–508 | 166–881 | 400 |
| unclean | 138 | ambulance=16, kamyun=16, kamyunet=16, minibus=20, neysan=51, savari=8, taxi=5, vanet=6 | 135–476 | 180–938 | 138 |

Quality findings:

| Finding | Count |
|---|---|
| none | 0 |

All 938 source files decode successfully, use JPEG format, and have the same eight classes except for `unclean/neysan`. Dimensions vary substantially (as expected for traffic-camera crops), but no image is unusually small under the declared rule. Statistical 1.5×IQR width/height review counts are recorded per split in `summary.json`; these are review flags, not automatic exclusions.

## Canonical-pixel duplicate findings

- Encoded-file SHA-256 unique values: **915**; duplicate groups: **23**.
- Canonical RGB pixel unique values: **915**; duplicate groups: **23**; extra copies: **23**.
- Cross-split groups: **23**; label conflicts: **1**; train–test overlap groups: **0** (the workflow fails closed if nonzero).

| Group | Members | Conflict? |
|---|---|---|
| 1 | train/ambulance/199984667.jpg; unclean/ambulance/199984667.jpg | no |
| 2 | train/ambulance/206808986.jpg; unclean/ambulance/206808986.jpg | no |
| 3 | train/ambulance/208738861.jpg; unclean/ambulance/208738861.jpg | no |
| 4 | train/ambulance/213342200.jpg; unclean/ambulance/213342200.jpg | no |
| 5 | train/ambulance/213801906.jpg; unclean/ambulance/213801906.jpg | no |
| 6 | train/ambulance/216216403.jpg; unclean/ambulance/216216403.jpg | no |
| 7 | train/ambulance/218591595.jpg; unclean/ambulance/218591595.jpg | no |
| 8 | train/ambulance/218966979.jpg; unclean/ambulance/218966979.jpg | no |
| 9 | train/kamyunet/193903806.jpg; unclean/kamyunet/193903806.jpg | no |
| 10 | train/kamyunet/194270924.jpg; unclean/kamyunet/194270924.jpg | no |
| 11 | train/kamyunet/195578789.jpg; unclean/kamyunet/195578789.jpg | no |
| 12 | train/kamyunet/217156552.jpg; unclean/kamyunet/217156552.jpg | no |
| 13 | train/minibus/198331521.jpg; unclean/minibus/198331521.jpg | no |
| 14 | train/minibus/205737488.jpg; unclean/minibus/205737488.jpg | no |
| 15 | train/vanet/214844236.jpg; unclean/neysan/214844236.jpg | yes |
| 16 | test/ambulance/200396165.jpg; unclean/ambulance/200396165.jpg | no |
| 17 | test/ambulance/214830125.jpg; unclean/ambulance/214830125.jpg | no |
| 18 | test/ambulance/215713763.jpg; unclean/ambulance/215713763.jpg | no |
| 19 | test/ambulance/216095686.jpg; unclean/ambulance/216095686.jpg | no |
| 20 | test/ambulance/217040489.jpg; unclean/ambulance/217040489.jpg | no |
| 21 | test/ambulance/218089422.jpg; unclean/ambulance/218089422.jpg | no |
| 22 | test/minibus/218138154.jpg; unclean/minibus/218138154.jpg | no |
| 23 | test/minibus/218307381.jpg; unclean/minibus/218307381.jpg | no |

Filename equality alone was not used. For example, `214844236.jpg` is a pixel-identical `train/vanet` and `unclean/neysan` image, demonstrating the apparent label conflict. It is excluded from the cleaned `unclean` analysis rather than relabelled or used for training.

## Cleaning decisions

| Dataset | Excluded | Clean retained |
|---|---|---|
| train | 0 | 400 |
| test (frozen) | 0 | 400 |
| unclean | 23 | 115 |

Retained `unclean` class counts: | Class | Count |
|---|---|
| ambulance | 2 |
| kamyun | 16 |
| kamyunet | 12 |
| minibus | 16 |
| neysan | 50 |
| savari | 8 |
| taxi | 5 |
| vanet | 6 |.

Every exclusion is recorded in `artifacts/audit/cleaned_unclean_exclusions.json`. All exclusions are duplicate copies inside `unclean`; none belongs to the original train or test split. This avoids leakage without silently changing the frozen evaluation sample.

## Representative examples

The reproducible 16-image gallery (two per known class) is saved as `reports/figures/representative_images.png` from these exact paths:

- `dataset/train/ambulance/198727855.jpg`
- `dataset/train/ambulance/199659807.jpg`
- `dataset/train/autobus/193229239.jpg`
- `dataset/train/autobus/194279229.jpg`
- `dataset/train/kamyun/196530974.jpg`
- `dataset/train/kamyun/198661910.jpg`
- `dataset/train/kamyunet/193903806.jpg`
- `dataset/train/kamyunet/194270924.jpg`
- `dataset/train/minibus/195315678.jpg`
- `dataset/train/minibus/196234914.jpg`
- `dataset/train/savari/193865576.jpg`
- `dataset/train/savari/195839967.jpg`
- `dataset/train/taxi/197362638.jpg`
- `dataset/train/taxi/198048598.jpg`
- `dataset/train/vanet/194285934.jpg`
- `dataset/train/vanet/194840733.jpg`
