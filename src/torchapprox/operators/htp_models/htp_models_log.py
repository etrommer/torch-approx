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
    res = 0.8039261609192951 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 0.9751701011185159 * base_func(op1, torch.ones_like(op2), **kwargs)
    res += 0.38615258688222637 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 9.70026217872455e-06 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res += 1.1141452728874457e-06 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_mitch_trunc_4(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_mitch_trunc_4
    """
    res = 0.8836583255216307 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 0.688865166865664 * base_func(op1, torch.ones_like(op2), **kwargs)
    res += 0.5393559127474923 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 1.6206246982543304e-05 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res += 2.4443187188374904e-05 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_mitch_trunc_5(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_mitch_trunc_5
    """
    res = 0.924461764098454 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 0.7440225215694504 * base_func(op1, torch.ones_like(op2), **kwargs)
    res += 0.8909653719790129 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 1.8190078013985422e-06 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res += 2.973607718928517e-05 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_drum_3(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_drum_3
    """
    res = 0.9955372239339377 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 0.8731411372391839 * base_func(op1, torch.ones_like(op2), **kwargs)
    res += 0.558677123646727 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 1.8913218498284312e-06 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res -= 2.6223821767604183e-05 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_drum_4(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_drum_4
    """
    res = 0.9988974792404028 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 0.44307824895118375 * base_func(op1, torch.ones_like(op2), **kwargs)
    res += 0.44862364644328057 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 1.631648608066416e-05 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res -= 1.8493145625686491e-06 * base_func(torch.ones_like(op1), op2**2, **kwargs)
    return res


def lin_drum_5(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication HTP Model for lin_drum_5
    """
    res = 0.9997177457049312 * base_func(op1, op2, **kwargs)
    res -= 0.0 * 1
    res += 0.4836883596074992 * base_func(op1, torch.ones_like(op2), **kwargs)
    res += 0.7260573667586884 * base_func(torch.ones_like(op1), op2, **kwargs)
    res += 4.390213316413094e-06 * base_func(op1**2, torch.ones_like(op2), **kwargs)
    res += 4.9343034296475685e-06 * base_func(torch.ones_like(op1), op2**2, **kwargs)
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
