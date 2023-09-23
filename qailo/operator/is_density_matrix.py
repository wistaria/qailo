from ..is_equal import is_equal
from .is_semi_positive import is_semi_positive
from .trace import trace


def is_density_matrix(op):
    return is_equal(trace(op), 1) and is_semi_positive(op)
