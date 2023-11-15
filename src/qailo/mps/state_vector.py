from .type import is_mps


def state_vector(m):
    assert is_mps(m)
    return m._state_vector()
