import numpy as np

from ..num_qubits import num_qubits
from ..util.letters import letters


def hconj(op):
    n = num_qubits(op)
    ss_from = letters()[: 2 * n]
    ss_to = ss_from[n : 2 * n] + ss_from[:n]
    return np.einsum("{}->{}".format(ss_from, ss_to), op).conjugate()
