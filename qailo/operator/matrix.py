def matrix(op):
    n = len(op.shape) // 2
    assert(op.shape == [2 for _ in range(2*n)])
    return op.reshape([2**n, 2**n])
