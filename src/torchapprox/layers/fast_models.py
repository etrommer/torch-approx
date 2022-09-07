"""
Template implementation for approximate multiplier fast models
"""


def mul8s_1l2d(base_func, op1, op2, kwargs):
    """
    Approximate Multiplication fast model for mul8s_1L2D

    Args:
        base_func: Basic layer function
        op1: 1st operand
        op2: 2nd operand
        kwargs: Keyword arguments passed to basic layer function
            (i.e. configuration parameters for Conv2d)

    Returns:
        Fast Model output for mul8s_1L2D
    """
    res = base_func(op1, op2, **kwargs)
    res -= base_func(op1, op2 % 4.0, **kwargs)
    res -= base_func(op1 % 4.0, op2, **kwargs)
    res += base_func(op1 % 4.0, op2 % 4.0, **kwargs)
    return res


def accurate(base_func, op1, op2, kwargs):
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
        Accurate Fast model output
    """
    return base_func(op1, op2, **kwargs)


fast_models = {
    "mul8s_1l2d": mul8s_1l2d,
    "accurate": accurate,
}
