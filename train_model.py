from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import (
    build_eval_transform,
    create_model,
    default_image_size_for_arch,
    save_vegetable_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a high-accuracy vegetable classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "Vegetable Images")
    parser.add_argument("--output", type=Path, default=Path("model") / "vegetable_cnn.pt")
    parser.add_argument("--arch", type=str, default="mobilenet_v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(data_dir: Path, image_size: int, batch_size: int, num_workers: int):
    transform = build_eval_transform(image_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=transform)
    val_ds = datasets.ImageFolder(data_dir / "validation", transform=transform)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


@torch.no_grad()
def extract_features(model: nn.Module, arch: str, loader: DataLoader, device: torch.device):
    model.eval()

    if arch in {"mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0"}:
        feature_net = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()).to(device)
    elif arch == "resnet18":
        feature_net = nn.Sequential(*list(model.children())[:-1], nn.Flatten()).to(device)
    else:
        raise ValueError(f"Feature extraction is not supported for arch: {arch}")

    all_features: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        feats = feature_net(inputs).cpu().numpy()
        all_features.append(feats)
        all_targets.append(targets.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_targets, axis=0)


@torch.no_grad()
def predict_with_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds: list[int] = []
    truths: list[int] = []
    for inputs, targets in loader:
        outputs = model(inputs.to(device))
        preds.extend(outputs.argmax(dim=1).cpu().tolist())
        truths.extend(targets.tolist())
    return np.array(truths), np.array(preds)


def save_classification_report(report_dict: dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1-score", "support"])
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                writer.writerow([
                    label,
                    metrics.get("precision", ""),
                    metrics.get("recall", ""),
                    metrics.get("f1-score", ""),
                    metrics.get("support", ""),
                ])


def generate_report_plots(report_dict: dict, val_acc: float, test_acc: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [
        label
        for label in report_dict
        if label not in {"accuracy", "macro avg", "weighted avg"}
    ]
    recalls = [float(report_dict[label].get("recall", 0.0)) for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot([1], [val_acc], marker="o", markersize=10, linewidth=2, label="Validation Accuracy")
    axes[0].plot([1], [test_acc], marker="o", markersize=10, linewidth=2, label="Test Accuracy")
    axes[0].set_title("Training Summary")
    axes[0].set_xlabel("Run")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(["Current model"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar(labels, recalls, color="seagreen")
    axes[1].set_title("Per-Class Detection Accuracy")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Recall")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(12, 5))
    plt.bar(labels, recalls, color="royalblue")
    plt.title("Per-Class Detection Accuracy (Test Set)")
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_accuracy.png", dpi=160, bbox_inches="tight")
    plt.close()


def set_linear_head_from_logreg(model: nn.Module, arch: str, clf: LogisticRegression) -> None:
    weight = torch.tensor(clf.coef_, dtype=torch.float32)
    bias = torch.tensor(clf.intercept_, dtype=torch.float32)

    if arch == "mobilenet_v2":
        model.classifier[1].weight.data.copy_(weight)
        model.classifier[1].bias.data.copy_(bias)
        return
    if arch == "mobilenet_v3_large":
        model.classifier[3].weight.data.copy_(weight)
        model.classifier[3].bias.data.copy_(bias)
        return
    if arch == "efficientnet_b0":
        model.classifier[1].weight.data.copy_(weight)
        model.classifier[1].bias.data.copy_(bias)
        return
    if arch == "resnet18":
        model.fc.weight.data.copy_(weight)
        model.fc.bias.data.copy_(bias)
        return

    raise ValueError(f"Unsupported arch for linear-head export: {arch}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.arch not in {"mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0", "resnet18"}:
        raise ValueError("Use a pretrained backbone arch for high accuracy, e.g. mobilenet_v2")

    image_size = args.image_size or default_image_size_for_arch(args.arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    class_names = train_ds.classes
    model = create_model(args.arch, num_classes=len(class_names), pretrained=True).to(device)

    print(f"Extracting features on {device} | arch={args.arch} | image_size={image_size}")
    print(f"Train={len(train_ds)} Validation={len(val_ds)} Test={len(test_ds)}")

    x_train, y_train = extract_features(model, args.arch, train_loader, device)
    x_val, y_val = extract_features(model, args.arch, val_loader, device)
    x_test, y_test = extract_features(model, args.arch, test_loader, device)

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=args.seed,
    )
    clf.fit(x_train, y_train)

    val_pred = clf.predict(x_val)
    test_pred = clf.predict(x_test)
    val_acc = float(np.mean(val_pred == y_val))
    test_acc = float(np.mean(test_pred == y_test))

    set_linear_head_from_logreg(model, args.arch, clf)
    save_vegetable_checkpoint(
        model,
        class_names=class_names,
        image_size=image_size,
        path=args.output,
        arch=args.arch,
    )

    report = classification_report(
        y_test,
        test_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    reports_dir = Path("reports")
    save_classification_report(report, reports_dir / "classification_report.csv")
    generate_report_plots(report, val_acc, test_acc, reports_dir)

    summary_path = reports_dir / "summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = "\n".join(
        [
            f"Dataset: {args.data_dir.resolve()}",
            f"Model: {args.output.resolve()}",
            f"Architecture: {args.arch}",
            f"Training method: pretrained feature extraction + multinomial logistic regression",
            f"Validation accuracy: {val_acc:.4f}",
            f"Test accuracy: {test_acc:.4f}",
        ]
    )
    summary_path.write_text(summary, encoding="utf-8")

    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()
