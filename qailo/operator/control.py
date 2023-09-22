import numpy as np
from ..util.shape import shape
from .identity import identity
from .matrix import matrix
from .pauli import x, z

def controlled(u):
    m = len(u.shape) // 2
    assert u.shape == shape(2*m)
    op = np.zeros([2, 2**m, 2, 2**m])
    op[0,:,0,:] = matrix(identity(m))
    op[1,:,1,:] = matrix(u)
    return op.reshape(shape(2*(m+1)))

def cx(n = 2):
    assert n > 1
    op = x()
    for i in range(n-1):
        op = controlled(op)
    return op

def cz(n = 2):
    assert n > 1
    op = z()
    for i in range(n-1):
        op = controlled(op)
    return op
