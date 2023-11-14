import torch


def htp_mitchell_trunc(base_func, op1, op2, kwargs):
    EPS = torch.tensor([1e-6])
    k = 3
    a = torch.floor(torch.log2(torch.maximum(op1, EPS)))
    b = torch.floor(torch.log2(torch.maximum(op2, EPS)))
    a = torch.maximum(2 ** (a - k + 1), torch.tensor([1]))
    b = torch.maximum(2 ** (b - k + 1), torch.tensor([1]))
    op1 -= op1 % a
    op2 -= op2 % b
    res = base_func(op1, op2, **kwargs)
    return res


def htp_drum(base_func, op1, op2, kwargs):
    EPS = torch.tensor([1e-6])
    k = 4
    a = torch.floor(torch.log2(torch.maximum(torch.abs(op1), EPS)))
    b = torch.floor(torch.log2(torch.maximum(torch.abs(op2), EPS)))
    a = torch.maximum(2 ** (a - k + 2), torch.tensor([1]))
    b = torch.maximum(2 ** (b - k + 2), torch.tensor([1]))

    op1 -= torch.fmod(op1, a)
    op2 -= torch.fmod(op2, b)

    a = torch.where(op1 > 0, a, -a)
    b = torch.where(op2 > 0, b, -b)

    # Debiasing
    op1 += a / 2
    op2 += b / 2

    res = base_func(op1, op2, **kwargs)
    return res
