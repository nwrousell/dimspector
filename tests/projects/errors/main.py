import torch
from signatures import (
    require_same_k,
    require_3d,
    require_2d,
    require_same_dim,
    with_expression,
    ShapeChecker,
)


def test_inconsistent_dimvars():
    x = torch.randn(32, 64)
    y = torch.randn(32, 128)
    return require_same_k(x, y)


def test_inconsistent_dimvars_1d():
    a = torch.randn(100)
    b = torch.randn(200)
    return require_same_dim(a, b)


def test_unequal_rank():
    x = torch.randn(32, 64, 128)
    return require_2d(x)


def test_matmul_mismatch():
    x = torch.randn(32, 64, 100)
    y = torch.randn(32, 200, 50)
    return x @ y


def test_broadcast_mismatch():
    x = torch.randn(32, 64)
    y = torch.randn(32, 100)
    return x + y


def test_bad_reshape():
    x = torch.randn(32, 64)
    return torch.reshape(x, (100, 30))


def test_signature_param_mismatch():
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)
    return with_expression(x, y)


def test_method_inconsistent():
    checker = ShapeChecker()
    x = torch.randn(16, 50)
    y = torch.randn(16, 75)
    return checker.check_dims(x, y)


def test_method_matmul():
    checker = ShapeChecker()
    x = torch.randn(8, 32, 64)
    y = torch.randn(8, 100, 16)
    return checker.matmul_wrapper(x, y)


if __name__ == "__main__":
    test_inconsistent_dimvars()
    test_inconsistent_dimvars_1d()
    test_unequal_rank()
    test_matmul_mismatch()
    test_broadcast_mismatch()
    test_bad_reshape()
    test_signature_param_mismatch()
    test_method_inconsistent()
    test_method_matmul()
