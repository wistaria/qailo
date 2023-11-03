import numpy as np

from .num_qubits import num_qubits


def is_canonical(m):
    if m.cp[1] - m.cp[0] < 2:
        for t in range(0, m.cp[0]):
            A = np.einsum("ijk,ijl->kl", m.tensors[t], m.tensors[t].conj())
            if not np.allclose(A, np.identity(A.shape[0])):
                return False
        for t in range(m.cp[1] + 1, num_qubits(m)):
            A = np.einsum("ijk,ljk->il", m.tensors[t], m.tensors[t].conj())
            if not np.allclose(A, np.identity(A.shape[0])):
                return False
        return True
    else:
        return False


def check_mps(m):
    """
    Check the shape of mps
    """
    n = num_qubits(m)

    # tensor shape
    dims = []
    assert m.tensors[0].shape[0] == 1
    dims.append(m.tensors[0].shape[0])
    for t in range(1, n - 1):
        dims.append(m.tensors[t].shape[0])
        assert m.tensors[t].shape[0] == m.tensors[t - 1].shape[2]
        assert m.tensors[t].shape[2] == m.tensors[t + 1].shape[0]
    assert m.tensors[n - 1].shape[2] == 1
    dims.append(m.tensors[n - 1].shape[0])
    dims.append(m.tensors[n - 1].shape[2])
    # print(dims)

    # qubit <-> tensor mapping
    for q in range(n):
        assert m.t2q[m.q2t[q]] == q
    for t in range(n):
        assert m.q2t[m.t2q[t]] == t

    # canonical position
    assert m.cp[0] in range(n)
    assert m.cp[1] in range(n)

    return True
