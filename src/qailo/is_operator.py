def is_operator(op):
    if op.shape[-1] == 2:
        n = len(op.shape) // 2
        assert op.shape == (2,) * (2 * n)
        return True
    else:
        return False
