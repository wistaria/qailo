import numpy as np

from ..is_operator import is_operator
from ..num_qubits import num_qubits
from ..operator.swap import swap


def apply_one(op, t):
    assert num_qubits(op) == 1
    return np.einsum("abc,bd->adc", t, op)


def apply_two(op, t0, t1, maxdim=None):
    assert num_qubits(op) == 2

    shape0 = t0.shape
    shape1 = t1.shape
    m = np.einsum("abc,cde,bdfg->afge", t0, t1, op).reshape((t0.shape[0] * 2, 2 * t1.shape[2]))
    U, S, V = np.linalg.svd(m, full_matrices=False)
    d = 0
    for i in range(len(S)):
        if S[i] > 1e-12:
            d = i + 1
    if maxdim is not None:
        d = min(d, maxdim)
    return (U[:, :d] * np.sqrt(S[:d])).reshape((shape0[0], 2, d)), (
        V[:d, :].transpose() * np.sqrt(S[:d])
    ).transpose().reshape((d, 2, shape1[2]))


def apply(op, mps, pos, maxdim=None):
    assert is_operator(op) and len(pos) == num_qubits(op)
    tensors = mps[3]
    if num_qubits(op) == 1:
        tensors[pos[0]] = apply_one(op, tensors[pos[0]])
    elif num_qubits(op) == 2:
        r = range(pos[1] - 1, pos[0], -1) if pos[0] < pos[1] else range(pos[1], pos[0])
        for k in r:
            print("swap between {} and {}".format(k, k + 1))
            tensors[k], tensors[k + 1] = apply_two(swap(), tensors[k], tensors[k + 1], maxdim)
        k = pos[0] if pos[0] < pos[1] else pos[0] - 1
        print("op between {} and {}".format(k, k + 1))
        tensors[k], tensors[k + 1] = apply_two(op, tensors[k], tensors[k + 1], maxdim)
        for k in reversed(r):
            print("swap between {} and {}".format(k, k + 1))
            tensors[k], tensors[k + 1] = apply_two(swap(), tensors[k], tensors[k + 1], maxdim)
    else:
        assert False
    return [mps[0], mps[1], mps[2], tensors]
