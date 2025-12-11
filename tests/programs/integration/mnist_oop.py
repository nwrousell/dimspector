from typing import Generic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class T(Generic(str)): ...


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for MNIST classification."""

    def __init__(
        self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: T["batch 1 28 28"]) -> T["batch classes"]:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10
    hidden_size = 256

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer
    model = MLP(input_size=784, hidden_size=hidden_size, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Training on {len(train_dataset):,} samples, testing on {len(test_dataset):,} samples"
    )
    print("-" * 60)

    # Training loop
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        # Evaluate
        model.eval()
        total_test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

                total_test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        test_loss = total_test_loss / len(test_loader)
        test_acc = correct / total

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}%"
        )

    print("-" * 60)
    print(f"Final test accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
