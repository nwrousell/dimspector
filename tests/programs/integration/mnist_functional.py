from typing import Generic
import torch


class T(Generic(str)): ...


def init_weight_matrix(in_features: int["in"], out_features: int["out"]) -> T["out in"]:
    """Initialize a weight matrix with Kaiming initialization."""
    weight = torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
    weight.requires_grad_(True)
    return weight


def init_bias_vector(out_features: int["out"]) -> T["out"]:
    """Initialize a bias vector with zeros."""
    bias = torch.zeros(out_features)
    bias.requires_grad_(True)
    return bias


def linear(x: T["batch in"], weight: T["out in"], bias: T["out"]) -> T["batch out"]:
    """Apply a linear transformation: x @ weight.T + bias."""
    return x @ torch.transpose(weight) + bias


def forward(
    x: T["batch 1 28 28"],
    w1: T["hidden 784"],
    b1: T["hidden"],
    w2: T["hidden hidden"],
    b2: T["hidden"],
    w3: T["classes hidden"],
    b3: T["classes"],
) -> T["batch classes"]:
    """Forward pass through the MLP."""
    x = torch.view(x, (x.shape[0], -1))
    x = torch.nn.functional.relu(linear(x, w1, b1))
    x = torch.nn.functional.relu(linear(x, w2, b2))
    x = linear(x, w3, b3)
    return x


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10
    input_size = 784
    hidden_size = 256
    num_classes = 10
    num_train = 1000
    num_test = 200

    # Device setup
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # Initialize weights
    w1 = init_weight_matrix(input_size, hidden_size)
    b1 = init_bias_vector(hidden_size)
    w2 = init_weight_matrix(hidden_size, hidden_size)
    b2 = init_bias_vector(hidden_size)
    w3 = init_weight_matrix(hidden_size, num_classes)
    b3 = init_bias_vector(num_classes)

    # Move to device
    w1 = w1.to(device)
    b1 = b1.to(device)
    w2 = w2.to(device)
    b2 = b2.to(device)
    w3 = w3.to(device)
    b3 = b3.to(device)

    # Mock dataset (zeros for images, random labels)
    train_images = torch.zeros(num_train, 1, 28, 28)
    train_labels = torch.randint(0, num_classes, (num_train,))
    test_images = torch.zeros(num_test, 1, 28, 28)
    test_labels = torch.randint(0, num_classes, (num_test,))

    # Move data to device
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    # Optimizer
    optimizer = torch.optim.Adam([w1, b1, w2, b2, w3, b3], lr=learning_rate)

    # Training loop
    num_train_batches = (num_train + batch_size - 1) // batch_size
    num_test_batches = (num_test + batch_size - 1) // batch_size

    for epoch in range(1, epochs + 1):
        # Train
        total_train_loss = 0.0
        for i in range(num_train_batches):
            start = i * batch_size
            end = min(start + batch_size, num_train)
            images = train_images[start:end]
            labels = train_labels[start:end]

            optimizer.zero_grad()
            outputs = forward(images, w1, b1, w2, b2, w3, b3)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / num_train_batches

        # Evaluate
        total_test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(num_test_batches):
                start = i * batch_size
                end = min(start + batch_size, num_test)
                images = test_images[start:end]
                labels = test_labels[start:end]

                outputs = forward(images, w1, b1, w2, b2, w3, b3)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

                total_test_loss += loss.item()
                predicted = outputs.max(1)[1]
                correct += predicted.eq(labels).sum().item()
                total += labels.shape[0]

        test_loss = total_test_loss / num_test_batches
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
