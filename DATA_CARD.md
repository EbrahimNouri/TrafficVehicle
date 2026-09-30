# Dataset Card — Traffic Vehicle Crops

## Summary

The supplied dataset contains cropped, overhead traffic-camera images of vehicles from eight operational classes plus a separate `unclean/neysan` group used for data-quality and unseen-class analysis.

| Split | Classes | Images |
|---|---|---:|
| `train` | ambulance, autobus, kamyun, kamyunet, minibus, savari, taxi, vanet | 400 |
| `test` | same eight classes | 400 |
| `unclean` | same eight classes plus neysan | 450 |

Counts above are the audited values for the supplied copy and are verified programmatically rather than assumed by the model code.

## Collection context

- Images are vehicle crops from fixed traffic cameras and often retain part of a green detector box.
- Vehicles vary in scale, pose, illumination, occlusion, background, and apparent sharpness.
- Original train/test counts are exactly balanced at 50 images per class.
- Class names are operational/taxonomy labels used by the source project; no additional semantic definitions were supplied.

## Known quality and leakage findings

- All 1,250 files are readable JPEGs; none is smaller than 64 pixels on either side.
- Encoded-file SHA-256 and decoded RGB pixel hashing both find 24 cross-split duplicate groups in this supplied copy, all involving `unclean`; 23 preserve the label and one conflicts (`train/vanet` versus `unclean/neysan`). The decoded hash remains encoding-independent by design.
- Frozen train/test remain unchanged. Duplicate `unclean` copies are excluded from the separate data-quality evaluation.

## Intended use

Educational supervised image classification, controlled model comparison, uncertainty analysis, and human-review decision support. It is not suitable by itself for safety-critical autonomous decisions.

## Limitations and bias

- The sample is small and may not cover geography, weather, lighting, road type, camera quality, or all vehicle variants.
- Labels may encode local operational categories rather than fine-grained international vehicle standards.
- Crops and detector boxes can introduce position/scale shortcuts.
- `neysan` is intentionally unseen in model fitting and is not evidence for a ninth class.
- Confidence is not an out-of-distribution guarantee.

## Provenance and license

The repository does not include acquisition metadata or a license for the supplied images. Users must confirm permission and retention terms before redistribution. The dataset directory is excluded from Git.
