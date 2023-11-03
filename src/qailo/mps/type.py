def is_canonical(m):
    return m._is_canonical()


def is_mps(m):
    return hasattr(m, "tensors")


def num_qubits(m):
    return len(m.tensors)
