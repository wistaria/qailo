class MPS(object):
    def __init__(self):
        True


def is_canonical(m):
    return m._is_canonical()


def is_mps(m):
    return isinstance(m, MPS)


def num_qubits(m):
    return len(m.q2t)
