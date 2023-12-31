import numpy as np


def is_density_matrix(v):
    if isinstance(v, np.ndarray):
        return (
            v.shape[-1] == 1
            and v.shape[-2] == 1
            and v.shape[-3] > 1
            and len(v.shape) % 2 == 0
        )
    return False


def is_operator(v):
    if isinstance(v, np.ndarray):
        return v.shape[-1] > 1 and len(v.shape) % 2 == 0
    return False


def num_qubits(v):
    if is_density_matrix(v):
        return (len(v.shape) - 2) // 2
    elif is_operator(v):
        return len(v.shape) // 2
    raise ValueError
