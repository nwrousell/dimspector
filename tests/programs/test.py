from typing import Generic
import torch


class T(Generic(str)): ...


def matmul_2d(a: T["m k"], b: T["k n"]):
    c = a @ b
    return c


def matmul_2d_1d(a: T["m k"], b: T["k"]):
    c = a @ b
    return c


def matmul_1d_2d(a: T["k"], b: T["k n"]):
    c = a @ b
    return c


def batched_matmul(a: T["batch m k"], b: T["batch k n"]):
    c = a @ b
    return c


def broadcast_add(a: T["batch h w"], b: T["h w"]):
    c = a + b
    return c


def scalar_multiply(a: T["batch dim"], scalar: float):
    c = a * scalar
    return c

def relu(x: T["b c h w"]):
    z = torch.nn.functional.relu(x)
    return z

def rdx_sum(x: T["b d"]):
    z = torch.sum(x, dim=1)
    return z


# def broadcast_dim1(a: T["batch 1 dim"], b: T["batch dim"]):
#     c = a + b
#     return c


# def create_ones():
#     x = torch.ones(3, 4)
#     return x


# def create_zeros():
#     x = torch.zeros(2, 3, 4)
#     return x


# def reshape_tensor(a: T["batch seq d"]):
#     b = a.reshape(-1, -1)
#     return b


# def view_tensor(a: T["batch d"]):
#     b = a.view(-1, 1, -1)
#     return b


# def transpose_tensor(a: T["h w"]):
#     b = a.transpose(0, 1)
#     return b


# def unsqueeze_tensor(a: T["batch dim"]):
#     b = a.unsqueeze(1)
#     return b


# def squeeze_tensor(a: T["batch 1 dim"]):
#     b = a.squeeze(1)
#     return b


# def linear_layer(x: T["batch in_dim"], weight: T["in_dim out_dim"], bias: T["out_dim"]):
#     out = x @ weight
#     out = out + bias
#     return out


# def reshape_matmul_reshape(a: T["batch seq d"], weight: T["d d_out"]):
#     a_flat = a.reshape(-1, -1)
#     out_flat = a_flat @ weight
#     out = out_flat.reshape(-1, -1, -1)
#     return out


# def conditional_reshape(a: T["batch dim"], flatten: bool):
#     if flatten:
#         b = a.reshape(-1)
#     else:
#         b = a
#     return b


# def transpose_then_matmul(a: T["batch d seq"], weight: T["d d_out"]):
#     a_t = a.transpose(1, 2)
#     out = a_t @ weight
#     return out


# def multi_op_broadcast(a: T["batch h w"], b: T["h w"], c: T["1 w"]):
#     result = a + b
#     result = result * c
#     return result


# def high_dim_matmul(q: T["batch heads seq d"], k: T["batch heads d seq"]):
#     scores = q @ k
#     return scores


# def elementwise_ops(a: T["batch dim"], b: T["dim"]):
#     add_result = a + b
#     mul_result = a * b
#     sub_result = a - b
#     return add_result, mul_result, sub_result


# def normalization_like(x: T["batch dim"], mean: T["dim"], std: T["dim"]):
#     normalized = (x - mean) / std
#     return normalized


# def conditional_ops(a: T["batch dim"], b: T["dim dim"], use_matmul: bool):
#     if use_matmul:
#         result = a @ b
#     else:
#         result = a + b
#     return result


# def multi_step_transform(
#     x: T["batch seq d"], w1: T["d d_hidden"], w2: T["d_hidden d_out"], bias: T["d_out"]
# ):
#     x_flat = x.reshape(-1, -1)
#     hidden = x_flat @ w1
#     out_flat = hidden @ w2
#     out_flat = out_flat + bias
#     out = out_flat.reshape(-1, -1, -1)
#     return out


# def matmul_chain(a: T["batch m k"], b: T["k n"], c: T["n p"], d: T["p q"]):
#     ab = a @ b
#     abc = ab @ c
#     abcd = abc @ d
#     return abcd


# def transpose_matmul_transpose(a: T["batch seq d"], weight: T["d d_out"]):
#     out = a @ weight
#     out_t = out.transpose(1, 2)
#     return out_t


# def constrained_reshape_ops(x: T["batch h w"], y: T["batch h w"]):
#     x_flat = x.reshape(-1, -1)
#     y_flat = y.reshape(-1, -1)
#     sum_flat = x_flat + y_flat
#     result = sum_flat.reshape(-1, -1, -1)
#     return result


# def multi_branch_constraints(
#     x: T["batch d"], w1: T["d d1"], w2: T["d d2"], use_first: bool
# ):
#     if use_first:
#         result = x @ w1
#     else:
#         result = x @ w2
#     return result


# def attention_pattern(
#     q: T["batch heads seq d"], k: T["batch heads seq d"], v: T["batch heads seq d"]
# ):
#     k_t = k.transpose(2, 3)
#     scores = q @ k_t
#     scores = scores * 0.125
#     output = scores @ v
#     return output


# def layer_norm_pattern(x: T["batch seq d"], gamma: T["d"], beta: T["d"]):
#     normalized = x
#     scaled = normalized * gamma
#     shifted = scaled + beta
#     return shifted


# def complex_reshape_chain(
#     x: T["batch seq d"], w1: T["d d1"], w2: T["d1 d2"], w3: T["d2 d"]
# ):
#     x_flat = x.reshape(-1, -1)
#     h1 = x_flat @ w1
#     h2 = h1 @ w2
#     h3 = h2 @ w3
#     out = h3.reshape(-1, -1, -1)
#     return out


# def broadcast_chain(a: T["batch h w"], b: T["h w"], c: T["1 w"], d: T["batch 1 1"]):
#     step1 = a + b
#     step2 = step1 * c
#     step3 = step2 * d
#     return step3


# def conditional_reshape_constraints(x: T["batch d"], flatten: bool, w: T["d out_dim"]):
#     if flatten:
#         x_reshaped = x.reshape(-1)
#         return x_reshaped
#     else:
#         result = x @ w
#         return result


# def multi_input_fusion(
#     x1: T["batch d1"], x2: T["batch d2"], w1: T["d1 d_out"], w2: T["d2 d_out"]
# ):
#     out1 = x1 @ w1
#     out2 = x2 @ w2
#     fused = out1 + out2
#     return fused


# def dimension_manipulation_chain(x: T["batch d"]):
#     x_expanded = x.unsqueeze(1)
#     x_reshaped = x_expanded.transpose(1, 2)
#     x_squeezed = x_reshaped.squeeze(2)
#     return x_squeezed


# def error_matmul_dimension_mismatch(a: T["m k"], b: T["n p"]):
#     c = a @ b
#     return c


# def error_broadcast_incompatible_shapes(a: T["batch h w"], b: T["h w c"]):
#     c = a + b
#     return c


# def error_matmul_chain_broken(a: T["batch m k"], b: T["k n"], c: T["p q"]):
#     ab = a @ b
#     abc = ab @ c
#     return abc


# def error_bias_shape_mismatch(
#     x: T["batch in_dim"], weight: T["in_dim out_dim"], bias: T["wrong_dim"]
# ):
#     out = x @ weight
#     out = out + bias
#     return out


# def error_batched_matmul_batch_mismatch(a: T["batch1 m k"], b: T["batch2 k n"]):
#     c = a @ b
#     return c


# def error_linear_layer_dimension_mismatch(
#     x: T["batch in_dim"], weight: T["wrong_dim out_dim"], bias: T["out_dim"]
# ):
#     out = x @ weight
#     out = out + bias
#     return out


# def error_broadcast_dim1_mismatch(a: T["batch 1 dim"], b: T["batch other_dim"]):
#     c = a + b
#     return c


# def error_multi_step_dimension_mismatch(
#     x: T["batch seq d"], w1: T["d d_hidden"], w2: T["wrong_dim d_out"], bias: T["d_out"]
# ):
#     x_flat = x.reshape(-1, -1)
#     hidden = x_flat @ w1
#     out_flat = hidden @ w2
#     out_flat = out_flat + bias
#     out = out_flat.reshape(-1, -1, -1)
#     return out


# def error_attention_dimension_mismatch(
#     q: T["batch heads seq d"],
#     k: T["batch heads wrong_d seq"],
#     v: T["batch heads seq d"],
# ):
#     k_t = k.transpose(2, 3)
#     scores = q @ k_t
#     output = scores @ v
#     return output


# def error_fusion_output_mismatch(
#     x1: T["batch d1"], x2: T["batch d2"], w1: T["d1 d_out1"], w2: T["d2 d_out2"]
# ):
#     out1 = x1 @ w1
#     out2 = x2 @ w2
#     fused = out1 + out2
#     return fused
