import torch


def accurate_reference(base_func, op1, op2, kwargs):
    """
    Accurate Multiplication
    """
    res = base_func(op1, op2, **kwargs)
    return res


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


def lin_mitch_trunc_3(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_mitch_trunc_3
    """
    res = -291628.54976695654 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res -= 844365.7988135596 * base_func(op1, torch.ones_like(op2), **kwargs)
    res += 844681.757549267 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 145927.21616702923 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res += 145702.14569062175 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_mitch_trunc_4(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_mitch_trunc_4
    """
    res = -54688.26160776692 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 229812.0945027578 * base_func(op1, torch.ones_like(op2), **kwargs)
    res -= 229541.34334737263 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 27394.30548518151 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res += 27294.85376210077 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_mitch_trunc_5(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_mitch_trunc_5
    """
    res = -39353.80716715142 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 252599.43194755862 * base_func(op1, torch.ones_like(op2), **kwargs)
    res -= 252333.63642277798 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 19724.117598738172 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res += 19630.627721638943 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_drum_3(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_drum_3
    """
    res = -128531.93374262762 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res -= 691992.4684840027 * base_func(op1, torch.ones_like(op2), **kwargs)
    res += 692066.8776468971 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 64313.92423645337 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res += 64218.99084463983 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_drum_4(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_drum_4
    """
    res = 30169.550095709164 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 101108.13886117878 * base_func(op1, torch.ones_like(op2), **kwargs)
    res -= 101104.9162441315 * base_func(torch.ones_like(op1), op2, **kwargs)
    res -= 15089.44349206188 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res -= 15079.11214436817 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_drum_5(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_drum_5
    """
    res = 8938.633726643624 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 2532.59179736041 * base_func(op1, torch.ones_like(op2), **kwargs)
    res -= 2538.4910649947997 * base_func(torch.ones_like(op1), op2, **kwargs)
    res -= 4470.955211496761 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res -= 4466.680050632531 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


htp_models_mul8s = {
    "accurate": accurate_reference,
    "htp_mitchell_trunc": htp_mitchell_trunc,
    "htp_drum": htp_drum,
    "lin_drum_3": lin_drum_3,
    "lin_drum_4": lin_drum_4,
    "lin_drum_5": lin_drum_5,
    "lin_mitch_trunc_3": lin_mitch_trunc_3,
    "lin_mitch_trunc_4": lin_mitch_trunc_4,
    "lin_mitch_trunc_5": lin_mitch_trunc_5,
}
