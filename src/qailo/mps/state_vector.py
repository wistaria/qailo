import numpy as np

from ..util.strops import letters, replace
from .num_qubits import num_qubits
from .type import is_mps


def state_vector(m):
    assert is_mps(m)
    n = num_qubits(m)
    v = m.tensors[0]
    for t in range(1, n):
        ss_v0 = letters()[: t + 2]
        ss_v1 = letters()[t + 1 : t + 4]
        ss_to = letters()[: t + 1] + letters()[t + 2 : t + 4]
        v = np.einsum(f"{ss_v0},{ss_v1}->{ss_to}", v, m.tensors[t])
    v = v.reshape((2,) * n)
    ss_from = letters()[:n]
    ss_to = ss_from
    for p in range(n):
        ss_to = replace(ss_to, p, ss_from[m.q2t[p]])
    return np.einsum(f"{ss_from}->{ss_to}", v).reshape((2,) * n + (1,))
