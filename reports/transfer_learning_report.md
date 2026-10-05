# ResNet18 Transfer Learning

Both transfer experiments retain the standard ImageNet ResNet18 (`conv1` and `maxpool` unchanged), ImageNet normalization, and resize/crop geometry. The training set has only 320 images, so validation—not test—controls the comparison. Total training budget is equal at 300 epochs. Recorded stages are feature extraction (head=300) and fine-tuning (head=100, layer4=200); the latter uses a smaller backbone learning rate after the head-only stage. Frozen BatchNorm statistics remain in evaluation mode.

| Strategy | Total params | Head-stage trainable | Head-stage frozen | % trainable | Initial LR groups | Best epoch |
|---|---|---|---|---|---|---|
| ResNet18 feature extraction | 11,180,616 | 4,104 | 11,176,512 | 0.04% | head=0.0005 | 168 |
| ResNet18 fine-tuning | 11,180,616 | 4,104 | 11,176,512 | 0.04% | head=0.0005 | 297 |

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
| 28 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 29 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 30 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 31 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 32 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 33 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 34 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 35 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 36 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 37 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 38 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 39 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 40 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 41 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 42 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 43 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 44 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 45 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 46 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 47 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 48 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 49 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 50 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 51 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 52 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 53 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 54 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 55 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 56 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 57 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 58 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 59 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 60 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 61 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 62 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 63 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 64 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 65 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 66 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 67 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 68 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 69 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 70 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 71 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 72 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 73 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 74 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 75 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 76 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 77 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 78 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 79 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 80 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 81 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 82 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 83 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 84 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 85 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 86 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 87 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 88 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 89 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 90 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 91 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 92 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 93 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 94 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 95 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 96 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 97 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 98 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 99 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 100 | head | [{"group": "head", "lr": 0.0005}] | {"head": 4104, "total_trainable": 4104} |
| 101 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 102 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 103 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 104 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 105 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 106 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 107 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 108 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 109 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 110 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 111 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 112 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 113 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 114 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 115 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 116 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 117 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 118 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 119 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 120 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 121 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 122 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 123 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 124 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 125 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 126 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 127 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 128 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 129 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 130 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 131 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 132 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 133 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 134 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 135 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 136 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 137 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 138 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 139 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 140 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 141 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 142 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 143 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 144 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 145 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 146 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 147 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 148 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 149 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 150 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 151 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 152 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 153 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 154 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 155 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 156 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 157 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 158 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 159 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 160 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 161 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 162 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 163 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 164 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 165 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 166 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 167 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 168 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 169 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 170 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 171 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 172 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 173 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 174 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 175 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 176 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 177 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 178 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 179 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 180 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 181 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 182 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 183 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 184 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 185 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 186 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 187 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 188 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 189 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 190 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 191 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 192 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 193 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 194 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 195 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 196 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 197 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 198 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 199 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 200 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 201 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 202 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 203 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 204 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 205 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 206 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 207 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 208 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 209 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 210 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 211 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 212 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 213 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 214 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 215 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 216 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 217 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 218 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 219 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 220 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 221 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 222 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 223 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 224 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 225 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 226 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 227 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 228 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 229 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 230 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 231 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 232 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 233 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 234 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 235 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 236 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 237 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 238 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 239 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 240 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 241 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 242 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 243 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 244 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 245 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 246 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 247 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 248 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 249 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 250 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 251 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 252 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 253 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 254 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 255 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 256 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 257 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 258 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 259 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 260 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 261 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 262 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 263 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 264 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 265 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 266 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 267 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 268 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 269 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 270 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 271 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 272 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 273 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 274 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 275 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 276 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 277 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 278 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 279 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 280 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 281 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 282 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 283 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 284 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 285 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 286 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 287 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 288 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 289 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 290 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 291 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 292 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 293 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 294 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 295 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 296 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 297 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 298 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 299 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |
| 300 | layer4 | [{"group": "head", "lr": 0.0005}, {"group": "pretrained_backbone", "lr": 1e-05}] | {"head": 4104, "total_trainable": 8397832} |

- Feature extraction validation macro-F1: **0.9026**.
- Fine-tuning validation macro-F1: **0.9231**.

The best transfer configuration is selected on validation. Test is untouched until final model selection is recorded.
