# ResNet18 Transfer Learning

Both transfer experiments retain the standard ImageNet ResNet18 (`conv1` and `maxpool` unchanged), ImageNet normalization, and resize/crop geometry. The training set has only 320 images, so validation—not test—controls the comparison. Total training budget is equal at 25 epochs. Recorded stages are feature extraction (head=25) and fine-tuning (head=8, layer4=17); the latter uses a smaller backbone learning rate after the head-only stage. Frozen BatchNorm statistics remain in evaluation mode.

| Strategy | Total params | Head-stage trainable | Head-stage frozen | % trainable | Initial LR groups | Best epoch |
|---|---|---|---|---|---|---|
| ResNet18 feature extraction | 11,180,616 | 4,104 | 11,176,512 | 0.04% | head=0.001 | 25 |
| ResNet18 fine-tuning | 11,180,616 | 4,104 | 11,176,512 | 0.04% | head=0.001 | 24 |

Fine-tuning stage details:

| Epoch | Stage | LR groups | Trainable parameters |
|---|---|---|---|
| 1 | head | [{"group": "head", "lr": 0.001}] | {"head": 4104, "total_trainable": 4104} |
| 2 | head | [{"group": "head", "lr": 0.001}] | {"head": 4104, "total_trainable": 4104} |
| 3 | head | [{"group": "head", "lr": 0.001}] | {"head": 4104, "total_trainable": 4104} |
| 4 | head | [{"group": "head", "lr": 0.001}] | {"head": 4104, "total_trainable": 4104} |
| 5 | head | [{"group": "head", "lr": 0.001}] | {"head": 4104, "total_trainable": 4104} |
| 6 | head | [{"group": "head", "lr": 0.001}] | {"head": 4104, "total_trainable": 4104} |
| 7 | head | [{"group": "head", "lr": 0.001}] | {"head": 4104, "total_trainable": 4104} |
| 8 | head | [{"group": "head", "lr": 0.001}] | {"head": 4104, "total_trainable": 4104} |
| 9 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 10 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 11 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 12 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 13 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 14 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 15 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 16 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 17 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 18 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 19 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 20 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 21 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 22 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 23 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 24 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |
| 25 | layer4 | [{"group": "head", "lr": 0.001}, {"group": "pretrained_backbone", "lr": 0.0001}] | {"head": 4104, "total_trainable": 8397832} |

- Feature extraction validation macro-F1: **0.8336**.
- Fine-tuning validation macro-F1: **0.9281**.

The best transfer configuration is selected on validation. Test is untouched until final model selection is recorded.
