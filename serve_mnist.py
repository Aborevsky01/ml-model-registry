import requests
from pathlib import Path

import torch
from torchvision import datasets, transforms

REGISTRY_URL = "http://127.0.0.1:8000"


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    model_name = "mnist-cnn"

    resp = requests.get(
        f"{REGISTRY_URL}/models/{model_name}/latest",
        params={"stage": "PRODUCTION"},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Error getting latest model: {resp.status_code} {resp.text}")

    version_info = resp.json()
    print("Latest PROD version info:", version_info)

    artifact_path = Path(version_info["artifact_path"])
    model_file = artifact_path / "model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=10).to(device)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    image, label = test_ds[0]  # первый пример

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))  # добавляем batch dimension
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()

    print(f"True label: {label}, predicted: {pred}")


if __name__ == "__main__":
    main()
