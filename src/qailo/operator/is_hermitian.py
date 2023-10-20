import numpy as np

from ..operator.hconj import hconj


def is_hermitian(op):
    return np.allclose(op, hconj(op))
