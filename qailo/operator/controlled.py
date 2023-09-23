import numpy as np

from ..util.shape import shape
from .matrix import matrix
from .pauli import x, z

def controlled(u):
    m = len(u.shape) // 2
    n = m+1
    assert u.shape == shape(2*m)

    op = np.identity(2**n).reshape([2, 2**m, 2, 2**m])
    op[1,:,1,:] = matrix(u)
    return op.reshape(shape(2*n))

def cx(n = 2):
    op = x()
    for i in range(n-1):
        op = controlled(op)
    return op

def cz(n = 2):
    op = z()
    for i in range(n-1):
        op = controlled(op)
    return op
