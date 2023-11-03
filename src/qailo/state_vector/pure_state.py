import numpy as np

from . import type as sv


def pure_state(v):
    assert sv.is_state_vector(v)
    n = sv.num_qubits(v)
    w = (v / np.linalg.norm(v)).reshape((2,) * n)
    ss0 = list(range(n))
    ss1 = list(range(n, 2 * n))
    shape = (2,) * (2 * n) + (1, 1)
    return np.einsum(w, ss0, w.conj(), ss1).reshape(shape)
