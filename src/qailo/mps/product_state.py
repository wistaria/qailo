import numpy as np

from ..state_vector.type import num_qubits
from ..state_vector.vector import vector
from .svd import tensor_svd


def product_state(n, c=0):
    assert n > 0
    tensors = []
    for t in range(n):
        tensor = np.zeros((1, 2, 1))
        tensor[0, (c >> (n - t - 1)) & 1, 0] = 1
        tensors.append(tensor)
    return tensors


def tensor_decomposition(v, nkeep=None, tol=1e-12):
    n = num_qubits(v)
    vv = vector(v).reshape((1, 2**n))
    tensors = []
    for t in range(n - 1):
        dims = vv.shape
        vv = vv.reshape(dims[0], 2, dims[1] // 2)
        t, vv = tensor_svd(vv, [[0, 1], [2]], "left", nkeep, tol)
        tensors.append(t)
    tensors.append(vv.reshape(vv.shape + (1,)))
    return tensors
