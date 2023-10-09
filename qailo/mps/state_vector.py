import numpy as np

from ..state_vector.shape import shape
from ..util.letters import letters


def state_vector(mps):
    pos, tensors = mps
    n = len(tensors)
    v = tensors[pos[0]]
    for i in range(1, n):
        ss_v0 = letters()[: i + 2]
        ss_v1 = letters()[i + 1 : i + 4]
        ss_to = letters()[: i + 1] + letters()[i + 2 : i + 4]
        v = np.einsum("{},{}->{}".format(ss_v0, ss_v1, ss_to), v, tensors[pos[i]])
    return v.reshape(shape(n))
