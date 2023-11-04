from . import type as sv


def vector(v, c=None):
    assert sv.is_state_vector(v)
    if c is None:
        n = sv.num_qubits(v)
        return v.reshape((2**n,))
    else:
        assert isinstance(c, list)
        return v[*c]
