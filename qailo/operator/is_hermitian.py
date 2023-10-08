import numpy as np

from ..operator.hconj import hconj


def is_hermitian(op):
    return (np.linalg.norm(op - hconj(op))) < 1e-15
