from .mps import MPS


def is_mps(m):
    return isinstance(m, MPS)
