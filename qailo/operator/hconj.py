import string
import numpy as np

from ..util.shape import shape

def hconj(op):
    n = len(op.shape) // 2
    assert(op.shape == shape(2*n))
    ss_from = string.ascii_lowercase[:2*n]
    ss_to = ss_from[n:2*n] + ss_from[0:n]
    return np.einsum("{}->{}".format(ss_from, ss_to), op).conjugate()
