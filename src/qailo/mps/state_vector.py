import numpy as np

from .type import is_mps, num_qubits


def state_vector(m):
    assert is_mps(m)
    n = num_qubits(m)
    v = m.tensors[0]
    for t in range(1, n):
        ss0 = list(range(t + 1)) + [t + 3]
        ss1 = [t + 3, t + 1, t + 2]
        v = np.einsum(v, ss0, m.tensors[t], ss1)
    v = v.reshape((2,) * n)
    return np.einsum(v, m.t2q).reshape((2,) * n + (1,))
