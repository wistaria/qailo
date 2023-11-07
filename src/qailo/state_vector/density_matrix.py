import numpy as np

from .type import is_state_vector, num_qubits


def density_matrix(v):
    assert is_state_vector(v)
    n = num_qubits(v)
    w = v.copy()
    w = (w / np.linalg.norm(w)).reshape((2,) * n)
    ss0 = list(range(n))
    ss1 = list(range(n, 2 * n))
    shape = (2,) * (2 * n) + (1, 1)
    return np.einsum(w, ss0, w.conj(), ss1).reshape(shape)
