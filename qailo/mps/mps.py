import numpy as np


def mps(n, c=0):
    assert n > 0
    tensors = []
    for i in range(n):
        t = np.zeros((1, 2, 1))
        t[0, (c >> (n - i - 1)) & 1, 0] = 1
        tensors.append(t)
    return [range(n), tensors]
