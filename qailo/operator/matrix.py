from ..is_density_matrix import is_density_matrix
from ..is_operator import is_operator
from ..num_qubits import num_qubits


def matrix(op):
    assert is_density_matrix(op) or is_operator(op)
    n = num_qubits(op)
    return op.reshape([2**n, 2**n])
