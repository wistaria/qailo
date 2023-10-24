import numpy as np

from ..util.letters import letters
from ..util.replace import replace
from .mps import tensor
from .type import is_mps, num_qubits


def state_vector(mps):
    assert is_mps(mps)
    n = num_qubits(mps)
    v = tensor(mps, 0)
    for t in range(1, n):
        ss_v0 = letters()[: t + 2]
        ss_v1 = letters()[t + 1 : t + 4]
        ss_to = letters()[: t + 1] + letters()[t + 2 : t + 4]
        v = np.einsum(f"{ss_v0},{ss_v1}->{ss_to}", v, tensor(mps, t))
    v = v.reshape((2,) * n)
    ss_from = letters()[:n]
    ss_to = ss_from
    for p in range(n):
        ss_to = replace(ss_to, p, ss_from[mps.q2t(p)])
    return np.einsum(f"{ss_from}->{ss_to}", v).reshape((2,) * n + (1,))
