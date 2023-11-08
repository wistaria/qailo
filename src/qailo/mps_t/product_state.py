from ..mps.product_state import tensor_decomposition
from ..state_vector.state_vector import one as sv_one
from ..state_vector.state_vector import zero as sv_zero
from .mps_t import MPS_T


def product_state(states):
    tensors = []
    for s in states:
        tensors = tensors + tensor_decomposition(s)
    return MPS_T(tensors)


def zero(n=1):
    return product_state([sv_zero()] * n)


def one(n=1):
    return product_state([sv_one()] * n)
