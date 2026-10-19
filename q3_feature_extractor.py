import csv
import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent


def create_loaders(batch_size=4, seed=0):
    data_dir = ROOT / "data/hymenoptera_data"
    if not (data_dir / "train").exists():
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        archive = data_dir.parent / "hymenoptera_data.zip"
        if not archive.exists():
            urllib.request.urlretrieve(
                "https://download.pytorch.org/tutorial/hymenoptera_data.zip", archive
            )
        with zipfile.ZipFile(archive) as stream:
            stream.extractall(data_dir.parent)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    trainset = datasets.ImageFolder(data_dir / "train", training_transform)
    valset = datasets.ImageFolder(data_dir / "val", validation_transform)
    if trainset.class_to_idx != valset.class_to_idx:
        raise ValueError("Training and validation classes must have the same order")
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0),
        "val": DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0),
    }
    return loaders


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": 100.0 * correct / len(loader.dataset),
        "correct": correct,
        "total": len(loader.dataset),
    }


def save_history(history, output_dir):
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_curves(history, output_dir):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for phase, label in [("train", "Training"), ("val", "Validation")]:
        axes[0].plot(epochs, [row[f"{phase}_loss"] for row in history], label=label)
        axes[1].plot(epochs, [row[f"{phase}_accuracy"] for row in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy loss", title="ResNet18 feature extractor loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="ResNet18 feature extractor accuracy")
    for axis in axes:
        axis.set_xticks([1, 5, 10, 15, 20, 25])
        axis.grid(alpha=0.2)
        axis.legend()
    fig.savefig(output_dir / "training_curve.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def save_predictions(model, loader, device, output_dir):
    model.eval()
    dataset = loader.dataset
    indices = [0, 1, 2] + [index for index, target in enumerate(dataset.targets) if target == 1][:3]
    images, targets = zip(*(dataset[index] for index in indices))
    batch = torch.stack(images)
    predictions = model(batch.to(device)).argmax(dim=1).cpu().tolist()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), layout="constrained")
    for axis, image, target, prediction in zip(axes.flat, images, targets, predictions):
        axis.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        axis.set_title(f"Actual: {dataset.classes[target]}\nPredicted: {dataset.classes[prediction]}")
        axis.axis("off")
    fig.savefig(output_dir / "validation_predictions.png", dpi=150)
    plt.close(fig)


def run_training(epochs=25, seed=0, verbose=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    os.environ["TORCH_HOME"] = str(ROOT / "data/torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_loaders(seed=seed)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights, progress=False)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(loaders["train"].dataset.classes))
    model = model.to(device)
    initial_features = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if not name.startswith("fc.")}
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    output_dir = ROOT / "results/q3_feature_extractor"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_correct = -1
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.eval()
        model.fc.train()
        train_loss = 0.0
        train_correct = 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
        scheduler.step()
        validation = evaluate(model, loaders["val"], criterion, device)
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss / len(loaders["train"].dataset),
            "train_accuracy": 100.0 * train_correct / len(loaders["train"].dataset),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        }
        history.append(row)
        save_history(history, output_dir)
        if validation["correct"] > best_correct:
            best_correct = validation["correct"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_weights.pt")
        if verbose:
            print(f"Epoch {epoch}: training {row['train_accuracy']:.2f}%, validation {row['val_accuracy']:.2f}%", flush=True)
    model.load_state_dict(torch.load(output_dir / "best_weights.pt", map_location=device, weights_only=True))
    selected = evaluate(model, loaders["val"], criterion, device)
    unchanged = all(torch.equal(value, model.state_dict()[name].cpu()) for name, value in initial_features.items())
    if not unchanged:
        raise RuntimeError("The frozen feature layers changed during training")
    result = {
        "model": "ResNet18",
        "pretrained_weights": "IMAGENET1K_V1",
        "mode": "Frozen feature extractor",
        "frozen_features_unchanged": unchanged,
        "dataset_url": "https://download.pytorch.org/tutorial/hymenoptera_data.zip",
        "classes": loaders["train"].dataset.classes,
        "train_images": len(loaders["train"].dataset),
        "validation_images": len(loaders["val"].dataset),
        "seed": seed,
        "device": str(device),
        "batch_size": 4,
        "optimizer": "SGD",
        "initial_learning_rate": 0.001,
        "momentum": 0.9,
        "scheduler_step": 7,
        "scheduler_gamma": 0.1,
        "training_metrics": "Online measurements with random training augmentation",
        "validation_transform": "Resize 256, center crop 224, ImageNet normalization",
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "selected_validation": selected,
        "final_metrics": history[-1],
        "history": history,
    }
    save_curves(history, output_dir)
    save_predictions(model, loaders["val"], device, output_dir)
    (output_dir / "training_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run_training()
