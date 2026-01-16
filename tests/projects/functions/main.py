import torch
from compute import (
    compute_matmul,
    compute_transpose,
    compute_broadcast,
    compute_concat,
)


def main():
    batch_size = 32
    m, k, n = 64, 128, 32
    h, w = 28, 28

    # Matrix multiplication
    x = torch.randn(batch_size, m, k)
    y = torch.randn(batch_size, k, n)
    result_matmul = compute_matmul(x, y)

    # Transpose
    img = torch.randn(batch_size, h, w)
    result_transpose = compute_transpose(img)

    # Broadcast
    vec = torch.randn(batch_size, k)
    bias = torch.randn(k)
    result_broadcast = compute_broadcast(vec, bias)

    # Concat
    x1 = torch.randn(batch_size, 64)
    x2 = torch.randn(batch_size, 32)
    result_concat = compute_concat(x1, x2)

    print(f"Matmul shape: {result_matmul.shape}")
    print(f"Transpose shape: {result_transpose.shape}")
    print(f"Broadcast shape: {result_broadcast.shape}")
    print(f"Concat shape: {result_concat.shape}")


if __name__ == "__main__":
    main()
