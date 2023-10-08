from .is_density_matrix import is_density_matrix
from .is_operator import is_operator
from .is_state_vector import is_state_vector


def num_qubits(q):
    if is_state_vector(q):
        return len(q.shape) - 1
    if is_density_matrix(q):
        return (len(q.shape) - 2) // 2
    if is_operator(q):
        return len(q.shape) // 2
    assert False
