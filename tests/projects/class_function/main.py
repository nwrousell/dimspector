from trainer import train_model, evaluate_model


def main():
    in_features = 784
    out_features = 10
    batch_size = 32

    # Train model using trainer function
    train_model(in_features, out_features, batch_size)

    # Evaluate model using trainer function
    # output = evaluate_model(model, batch_size, in_features)
    # print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
