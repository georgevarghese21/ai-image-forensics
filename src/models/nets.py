"""Models for V1. Every model outputs a single number per image."""
import torch
import torch.nn as nn
from torchvision import models


class ForensicCNN(nn.Module):
    """Small from-scratch baseline (~1.2M parameters).

    Stride-1 convolutions with pooling only at the end of each block.
    Standard classifier stems downsample immediately, discarding the
    high-frequency detail that generator artefacts live in.
    """

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class Squeeze(nn.Module):
    """Wraps a torchvision model so it returns shape (B,) not (B, 1)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_model(name, pretrained=True):
    """Factory so train.py can switch architecture from a config string."""
    name = name.lower()

    if name == "forensic_cnn":
        return ForensicCNN()

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, 1)
        return Squeeze(net)

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
        return Squeeze(net)

    raise ValueError(f"unknown model: {name}")
