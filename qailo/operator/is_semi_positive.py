import numpy as np

from .is_hermitian import is_hermitian
from .matrix import matrix


def is_semi_positive(op):
    if not is_hermitian(op):
        return False
    evs = np.linalg.eigvalsh(matrix(op))
    print(evs)
    for ev in evs:
        if ev < -1e-15:
            return False
    return True
