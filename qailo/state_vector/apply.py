import numpy as np

from ..is_state_vector import is_state_vector
from ..is_operator import is_operator
from ..num_qubits import num_qubits
from ..util.letters import letters
from ..util.replace import replace


def apply(op, sv, pos=None):
    assert is_operator(op) and is_state_vector(sv)
    n = num_qubits(sv)
    m = num_qubits(op)
    if pos is None:
        assert m == n
        pos = range(n)
    assert len(pos) == m

    ss_op = letters()[: 2 * m]
    ss_v = ss_to = letters()[2 * m : 2 * m + n + 1]
    for i in range(m):
        ss_v = replace(ss_v, pos[i], ss_op[m + i])
        ss_to = replace(ss_to, pos[i], ss_op[i])
    return np.einsum("{},{}->{}".format(ss_v, ss_op, ss_to), sv, op)
