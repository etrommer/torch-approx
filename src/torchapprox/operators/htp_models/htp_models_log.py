import torch


def htp_mitchell_trunc(base_func, op1, op2, kwargs):
    EPS = 1e-6
    k = 3
    a = torch.floor(torch.log2(torch.maximum(op1, EPS)))
    b = torch.floor(torch.log2(torch.maximum(op2, EPS)))
    a = torch.maximum(2 ** (a - k + 1), 1)
    b = torch.maximum(2 ** (b - k + 1), 1)
    op1 -= op1 % a
    op2 -= op2 % b
    res = base_func(op1, op2, **kwargs)
    return res


def htp_drum(base_func, op1, op2, kwargs):
    EPS = 1e-6
    k = 3
    a = torch.floor(torch.log2(torch.maximum(op1, EPS)))
    b = torch.floor(torch.log2(torch.maximum(op2, EPS)))
    a = torch.maximum(2 ** (a - k + 1), 1)
    b = torch.maximum(2 ** (b - k + 1), 1)
    op1 -= op1 % a
    op2 -= op2 % b

    # Debiasing
    op1 += torch.floor(a / 2)
    op2 += torch.floor(b / 2)

    res = base_func(op1, op2, **kwargs)
    return res
