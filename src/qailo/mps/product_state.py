from ..state_vector.state_vector import one as sv_one
from ..state_vector.state_vector import zero as sv_zero
from ..state_vector.type import num_qubits
from ..state_vector.vector import vector
from .mps_c import MPS_C
from .svd import tensor_svd
from .type import is_mps


def tensor_decomposition(v, nkeep=None, tol=1e-12):
    if is_mps(v):
        return v.tensors
    else:
        n = num_qubits(v)
        w = vector(v).reshape((1, 2**n))
        tensors = []
        for t in range(n - 1):
            dims = w.shape
            w = w.reshape(dims[0], 2, dims[1] // 2)
            t, w = tensor_svd(w, [[0, 1], [2]], "left", nkeep, tol)
            tensors.append(t)
        tensors.append(w.reshape(w.shape + (1,)))
        return tensors


def product_state(states, mps=MPS_C):
    tensors = []
    for s in states:
        tensors = tensors + tensor_decomposition(s)
    return mps(tensors)


def zero(n=1, mps=MPS_C):
    return product_state([sv_zero()] * n, mps)


def one(n=1, mps=MPS_C):
    return product_state([sv_one()] * n, mps)
