import numpy as np


def is_state_vector(v):
    if isinstance(v, np.ndarray):
        return v.shape[-1] == 1 and v.shape[-2] > 1
    else:
        return False


def num_qubits(v):
    return v.ndim - 1
