from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
)

DEFAULT_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
DEFAULT_STD: Sequence[float] = (0.229, 0.224, 0.225)

SUPPORTED_ARCHES = (
    "vegetable_cnn",
    "mobilenet_v3_large",
    "mobilenet_v2",
    "efficientnet_b0",
    "resnet18",
)


class VegetableCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def save_vegetable_checkpoint(
    model: nn.Module,
    class_names: Sequence[str],
    image_size: int,
    path: str | Path,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
    **extra: Any,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "image_size": int(image_size),
        "mean": list(mean),
        "std": list(std),
        "arch": extra.pop("arch", "vegetable_cnn"),
        **extra,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_vegetable_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 128))
    arch = str(checkpoint.get("arch", "vegetable_cnn")).lower()

    if arch == "vegetable_cnn":
        model = VegetableCNN(num_classes=len(class_names)).to(device)
    else:
        model = create_model(arch=arch, num_classes=len(class_names), pretrained=False).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return {
        "model": model,
        "class_names": class_names,
        "image_size": image_size,
        "device": str(device),
        "checkpoint": checkpoint,
        "arch": arch,
    }


def default_image_size_for_arch(arch: str) -> int:
    arch = arch.lower()
    if arch == "vegetable_cnn":
        return 128
    if arch == "mobilenet_v2":
        return 160
    return 224


def create_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    arch = arch.lower()
    if arch not in SUPPORTED_ARCHES:
        raise ValueError(
            f"Unsupported arch '{arch}'. Supported: {', '.join(SUPPORTED_ARCHES)}"
        )

    if arch == "vegetable_cnn":
        return VegetableCNN(num_classes=num_classes)

    if arch == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        try:
            model = models.mobilenet_v3_large(weights=weights)
        except Exception:
            model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    if arch == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        try:
            model = models.mobilenet_v2(weights=weights)
        except Exception:
            model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, arch: str, train_backbone: bool) -> None:
    if train_backbone:
        return

    for param in model.parameters():
        param.requires_grad = False

    for param in _head_parameters(model, arch):
        param.requires_grad = True


def _head_parameters(model: nn.Module, arch: str) -> Iterable[torch.nn.Parameter]:
    arch = arch.lower()
    if arch == "vegetable_cnn":
        return model.classifier.parameters()
    if arch == "mobilenet_v3_large":
        return model.classifier.parameters()
    if arch == "mobilenet_v2":
        return model.classifier.parameters()
    if arch == "efficientnet_b0":
        return model.classifier.parameters()
    return model.fc.parameters()


def trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = build_eval_transform(image_size)
    return transform(image.convert("RGB")).unsqueeze(0)
