import numpy as np

from ..state_vector.shape import shape
from ..util.letters import letters


def state_vector(mps):
    n = mps.num_qubits()
    v = mps.tensors_[0]
    for i in range(1, n):
        ss_v0 = letters()[: i + 2]
        ss_v1 = letters()[i + 1 : i + 4]
        ss_to = letters()[: i + 1] + letters()[i + 2 : i + 4]
        v = np.einsum("{},{}->{}".format(ss_v0, ss_v1, ss_to), v, mps.tensors_[i])
    return v.reshape(shape(n))
