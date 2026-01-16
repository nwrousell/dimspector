import torch
import torch.nn as nn
from model import MLP


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

    # Initialize model
    model = MLP(input_size, hidden_size, num_classes)
    model = model.to(device)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
            outputs = model.forward(images)
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

                outputs = model.forward(images)
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
