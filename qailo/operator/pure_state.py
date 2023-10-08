import numpy as np

from ..num_qubits import num_qubits
from ..is_state_vector import is_state_vector
from ..util.letters import letters


def pure_state(sv):
    assert is_state_vector(sv)
    n = num_qubits(sv)
    v = sv / np.linalg.norm(sv)
    str_from0 = letters()[: n + 1]
    str_from1 = letters()[n + 1 : 2 * n + 2]
    str_to = (
        letters()[:n]
        + letters()[n + 1 : 2 * n + 1]
        + letters()[n]
        + letters()[2 * n + 1]
    )
    return np.einsum("{},{}->{}".format(str_from0, str_from1, str_to), v, v.conj())
