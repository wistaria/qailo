from ..is_state_vector import is_state_vector
from ..num_qubits import num_qubits


def vector(sv):
    assert is_state_vector(sv)
    n = num_qubits(sv)
    return sv.reshape([2**n])
