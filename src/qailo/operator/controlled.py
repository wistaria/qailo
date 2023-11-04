import numpy as np

from .matrix import matrix
from .one_qubit import p, x, z
from .type import is_operator, num_qubits


def controlled(u):
    assert is_operator(u)
    m = num_qubits(u)
    n = m + 1
    op = np.identity(2**n).reshape([2, 2**m, 2, 2**m])
    op[1, :, 1, :] = matrix(u)
    return op.reshape((2,) * (2 * n))


def cp(phi):
    return controlled(p(phi))


def cx(n=2):
    op = x()
    for _ in range(n - 1):
        op = controlled(op)
    return op


def cz(n=2):
    op = z()
    for _ in range(n - 1):
        op = controlled(op)
    return op


# mpo representation of controlled gates
def control_begin():
    op = np.zeros([2, 2, 2, 2, 2])
    op[0, 0, 0, 0, 0] = 1
    op[0, 1, 0, 0, 1] = 1
    op[1, 0, 0, 1, 0] = 1
    op[1, 1, 1, 1, 1] = 1
    return op.reshape([2, 4, 2, 2])


def control_propagate():
    op = np.zeros([2, 2, 2, 2, 2, 2])
    op[0, 0, 0, 0, 0, 0] = 1
    op[1, 0, 0, 1, 0, 0] = 1
    op[0, 0, 0, 0, 1, 0] = 1
    op[1, 0, 0, 1, 1, 0] = 1
    op[0, 1, 0, 0, 0, 1] = 1
    op[1, 1, 0, 1, 0, 1] = 1
    op[0, 1, 1, 0, 1, 1] = 1
    op[1, 1, 1, 1, 1, 1] = 1
    return op.reshape([2, 4, 4, 2])


def control_end(u):
    assert is_operator(u)
    m = num_qubits(u)
    n = m + 1
    op = np.zeros([2, 2**m, 2, 2, 2**m])
    op[0, :, 0, 0, :] = np.identity(2**m)
    op[0, :, 0, 1, :] = u
    op[1, :, 1, 0, :] = np.identity(2**m)
    op[1, :, 1, 1, :] = u
    return op.reshape((2,) * n + (4,) + (2,) * m)
