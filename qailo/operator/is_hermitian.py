import numpy as np

from .hconj import hconj


def is_hermitian(op):
    return (np.linalg.norm(op - hconj(op))) < 1e-15
