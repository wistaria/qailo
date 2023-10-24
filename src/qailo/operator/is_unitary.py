from .type import is_operator, num_qubits
from .hconj import hconj
from .is_identity import is_identity
from .multiply import multiply


def is_unitary(op):
    assert is_operator(op)
    n = num_qubits(op)
    return is_identity(multiply(hconj(op), op, range(n))) and is_identity(
        multiply(op, hconj(op), range(n))
    )
