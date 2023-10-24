import numpy as np

from . import type as sv
from ..util.letters import letters


def pure_state(v):
    assert sv.is_state_vector(v)
    n = sv.num_qubits(v)
    v = v / np.linalg.norm(v)
    ss_from0 = letters()[: n + 1]
    ss_from1 = letters()[n + 1 : 2 * n + 2]
    ss_to = (
        letters()[:n]
        + letters()[n + 1 : 2 * n + 1]
        + letters()[n]
        + letters()[2 * n + 1]
    )
    return np.einsum("{},{}->{}".format(ss_from0, ss_from1, ss_to), v, v.conj())
