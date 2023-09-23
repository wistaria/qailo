import numpy as np

from ..util.letters import letters


def pure_state(sv):
    n = len(sv.shape)
    return np.einsum("{},{}".format(letters()[:n], letters()[n : 2 * n]), sv, sv.conj())
