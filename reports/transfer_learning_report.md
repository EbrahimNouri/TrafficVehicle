# ResNet18 Transfer Learning

Both transfer experiments retain the standard ImageNet ResNet18 (`conv1` and `maxpool` unchanged), ImageNet normalization, and resize/crop geometry. The training set has only 320 images, so validation—not test—controls the comparison. Total training budget is equal at 80 epochs. Recorded stages are feature extraction (head=80) and fine-tuning (head=27, layer4=53); the latter uses a smaller backbone learning rate after the head-only stage. Frozen BatchNorm statistics remain in evaluation mode.

| Strategy | Total params | Head-stage trainable | Head-stage frozen | % trainable | Initial LR groups | Best epoch |
|---|---|---|---|---|---|---|
| ResNet18 feature extraction | 11,180,616 | 4,104 | 11,176,512 | 0.04% | head=0.0005 | 45 |
| ResNet18 fine-tuning | 11,180,616 | 4,104 | 11,176,512 | 0.04% | head=0.0005 | 80 |

Fine-tuning stage details:

| Epoch | Stage | LR groups | Trainable parameters |
|---|---|---|---|
| 1 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 2 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 3 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 4 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 5 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 6 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 7 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 8 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 9 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 10 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 11 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 12 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 13 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 14 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 15 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 16 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 17 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 18 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 19 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 20 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 21 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 22 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 23 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 24 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 25 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 26 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 27 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 28 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 29 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 30 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 31 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 32 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 33 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 34 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 35 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 36 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 37 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 38 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 39 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 40 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 41 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 42 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 43 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 44 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 45 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 46 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 47 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 48 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 49 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 50 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 51 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 52 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 53 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 54 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 55 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 56 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 57 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 58 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 59 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 60 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 61 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 62 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 63 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 64 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 65 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 66 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 67 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 68 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 69 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 70 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 71 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 72 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 73 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 74 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 75 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 76 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 77 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 78 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 79 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 80 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |

- Feature extraction validation macro-F1: **0.8606**.
- Fine-tuning validation macro-F1: **0.9165**.

The best transfer configuration is selected on validation. Test is untouched until final model selection is recorded.
