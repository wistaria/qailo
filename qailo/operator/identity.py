import numpy as np

def identity(n):
    return np.identity(2**n).reshape([2 for _ in range(2*n)])
