def is_canonical(m):
    return m.is_canonical()


def is_mps(m):
    return hasattr(m, "tensors")
