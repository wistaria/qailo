from .hconj import hconj
from .is_identity import is_identity
from .multiply import multiply

def is_unitary(op):
    return is_identity(multiply(hconj(op), op))
