"""
Template implementation for approximate multiplier fast models
"""
import torch


def mul8s_1kv8(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication accurate fast model.
    This yields the same output as the baseline layer.
    Primarily useful for testing

    Args:
        base_func: Basic layer function
        op1: 1st operand
        op2: 2nd operand
        kwargs: Keyword arguments passed to basic layer function
            (i.e. configuration parameters for Conv2d)

    Returns:
        Accurate Layer operation result
    """
    return base_func(op1, op2, **kwargs)


def mul8s_1kr3(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1KR3
    """
    res = base_func(op1, op2, **kwargs)
    res -= base_func(op1, (op2 % 64.0), **kwargs)
    return res


def mul8s_1kr6(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1KR6
    """
    res = base_func(op1, op2, **kwargs)
    res -= base_func(op1, (op2 % 2.0), **kwargs)
    return res


def mul8s_1kr8(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1KR8
    """
    res = base_func(op1, op2, **kwargs)
    res -= base_func(op1, (op2 % 2.0), **kwargs)
    return res


def mul8s_1krc(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1KRC
    """
    res = base_func(op1, op2, **kwargs)
    res -= 3.0 * base_func((op1 % 2.0), torch.ones_like(op2), **kwargs)
    res -= base_func(op1, (op2 % 2.0), **kwargs)
    return res


def mul8s_1kty(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1KTY
    """
    res = base_func(op1, op2, **kwargs)
    res -= base_func(op1, (op2 % 8.0), **kwargs)
    return res


def mul8s_1kva(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1KVA
    """
    res = base_func(op1, op2, **kwargs)
    res -= 0.3 * base_func((op1 % 4.0), (op2 % 4.0), **kwargs)
    return res


def mul8s_1kvb(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1KVB
    """
    res = base_func(op1, op2, **kwargs)
    res -= base_func((op1 % 4.0), (op2 % 4.0), **kwargs)
    return res


def mul8s_1kvl(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1KVL
    """
    res = base_func(op1, op2, **kwargs)
    res -= 0.6 * base_func((op1 % 4.0), torch.ones_like(op2), **kwargs)
    res -= base_func(op1, (op2 % 4.0), **kwargs)
    res -= 2.0 * base_func((op1 % 4.0) ** 2, torch.ones_like(op2), **kwargs)
    res -= 0.1 * base_func((op1 % 4.0), (op2 % 4.0), **kwargs)
    return res


def mul8s_1kx2(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1KX2
    """
    res = base_func(op1, op2, **kwargs)
    res -= 0.1 * base_func((op1 % 8.0) ** 2, torch.ones_like(op2), **kwargs)
    res -= 0.5 * base_func((op1 % 8.0), (op2 % 8.0), **kwargs)
    res -= 0.1 * base_func(torch.ones_like(op1), (op2 % 8.0) ** 2, **kwargs)
    return res


def mul8s_1l1g(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1L1G
    """
    res = base_func(op1, op2, **kwargs)
    res -= base_func(op1, (op2 % 8.0), **kwargs)
    res -= base_func((op1 % 8.0), (op2), **kwargs)
    res += base_func((op1 % 8.0), (op2 % 8.0), **kwargs)
    return res


def mul8s_1l2d(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1L2D
    """
    res = base_func(op1, op2, **kwargs)
    res -= base_func(op1, (op2 % 4.0), **kwargs)
    res -= base_func((op1 % 4.0), (op2), **kwargs)
    res += 0.9 * base_func((op1 % 4.0), (op2 % 4.0), **kwargs)
    return res


def mul8s_1l2h(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication Fast Model for mul8s_1L2H
    """
    res = base_func(op1, op2, **kwargs)
    res -= base_func(op1, (op2 % 2.0), **kwargs)
    res -= base_func((op1 % 2.0), (op2), **kwargs)
    return res


fast_models = {
    "mul8s_1KR3": mul8s_1kr3,
    "mul8s_1KR6": mul8s_1kr6,
    "mul8s_1KR8": mul8s_1kr8,
    "mul8s_1KRC": mul8s_1krc,
    "mul8s_1KTY": mul8s_1kty,
    "mul8s_1KV8": mul8s_1kv8,
    "mul8s_1KVA": mul8s_1kva,
    "mul8s_1KVB": mul8s_1kvb,
    "mul8s_1KVL": mul8s_1kvl,
    "mul8s_1KX2": mul8s_1kx2,
    "mul8s_1L1G": mul8s_1l1g,
    "mul8s_1L2D": mul8s_1l2d,
    "mul8s_1L2H": mul8s_1l2h,
}
