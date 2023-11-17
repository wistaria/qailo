class mps(object):
    def __init__(self):
        True


def is_canonical(m):
    return m._is_canonical()


def is_mps(m):
    return isinstance(m, mps)


def num_qubits(m):
    return len(m.q2t)
