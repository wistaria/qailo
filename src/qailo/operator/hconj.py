import numpy as np

from ..util.strops import letters
from .type import is_operator, num_qubits


def hconj(op):
    assert is_operator(op)
    n = num_qubits(op)
    ss_from = letters()[: 2 * n]
    ss_to = ss_from[n : 2 * n] + ss_from[:n]
    return np.einsum("{}->{}".format(ss_from, ss_to), op).conjugate()
