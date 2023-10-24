import numpy as np

from .hconj import hconj


def is_hermitian(op):
    return np.allclose(op, hconj(op))
