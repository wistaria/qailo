from ..is_equal import is_equal
from ..util.shape import shape
from .identity import identity

def is_identity(op):
    n = len(op.shape) // 2
    return (op.shape == shape(2*n)) and is_equal(op, identity(n))
