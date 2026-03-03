import json
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

REGISTRY_URL = "http://127.0.0.1:8000"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),                            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                           
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 2 

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    model_name = "mnist-cnn"
    team = "mlds_180"
    version_dir = Path("models") / team / model_name / "v1" 
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    metrics = {
        "train_loss": float(train_loss),
        "train_accuracy": float(train_acc),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
    }
    metrics_path = version_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "model": "SimpleCNN",
        "num_classes": 10,
        "optimizer": "Adam",
        "lr": 1e-3,
        "batch_size": 64,
        "num_epochs": num_epochs,
    }
    config_path = version_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    artifact_path = str(version_dir)
    print(f"Saved model to {model_path}")
    print(f"Artifact path: {artifact_path}")

    resp = requests.post(
        f"{REGISTRY_URL}/models",
        json={
            "name": model_name,
            "description": "Simple CNN on MNIST (PyTorch)",
            "domain": "cv_mnist",
            "owner": "mlds_180_team",
        },
    )
    
    if resp.status_code not in (200, 201, 400):
        raise RuntimeError(f"Error creating model: {resp.status_code} {resp.text}")
    print("Model create response:", resp.status_code, resp.text)

    resp = requests.post(
        f"{REGISTRY_URL}/models/{model_name}/versions",
        json={
            "artifact_path": artifact_path,
            "git_commit": "dummy_mnist_commit",
            "data_ref": "torchvision.datasets.MNIST(train=True)",
            "params": config,
            "metrics": metrics,
            "created_by": "andrej",
            "training_env": str(device),
            "pipeline_version": "v1",
            "run_id": "mnist-run-001",
        },
    )
    print("Version create response:", resp.status_code, resp.text)
    resp.raise_for_status()
    print("Registered version:", resp.json())


if __name__ == "__main__":
    main()
