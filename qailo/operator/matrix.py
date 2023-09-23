from ..util.shape import shape

def matrix(op):
    n = len(op.shape) // 2
    assert(op.shape == shape(2*n))

    return op.reshape([2**n, 2**n])
