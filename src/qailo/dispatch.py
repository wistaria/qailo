from . import mps
from . import state_vector as sv


def apply(v, p, pos=None, maxdim=None):
    if sv.is_state_vector(v):
        v = sv.apply(v, p, pos)
    elif mps.is_mps(v):
        v = mps.apply(v, p, pos, maxdim)
    return v


def apply_seq(v, seq, maxdim=None):
    if sv.is_state_vector(v):
        v = sv.apply_seq(v, seq)
    elif mps.is_mps(v):
        v = mps.apply_seq(v, seq, maxdim)
    return v


def probability(v, c=None):
    if sv.is_state_vector(v):
        return sv.probability(v, c)
    elif mps.is_mps(v):
        return sv.probability(mps.state_vector(v), c)


def vector(v, c=None):
    if sv.is_state_vector(v):
        return sv.vector(v, c)
    elif mps.is_mps(v):
        return sv.vector(mps.state_vector(v), c)
