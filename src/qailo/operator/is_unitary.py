from .hconj import hconj
from .is_identity import is_identity
from .multiply import multiply


def is_unitary(op):
    n = len(op.shape) // 2
    return is_identity(multiply(hconj(op), op, range(n))) and is_identity(multiply(op, hconj(op), range(n)))
