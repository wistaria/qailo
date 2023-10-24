from .type import is_density_matrix, is_operator, num_qubits


def matrix(op):
    assert is_density_matrix(op) or is_operator(op)
    n = num_qubits(op)
    return op.reshape([2**n, 2**n])
