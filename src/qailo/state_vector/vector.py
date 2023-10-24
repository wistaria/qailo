from . import type as sv


def vector(v):
    assert sv.is_state_vector(v)
    n = sv.num_qubits(v)
    return v.reshape((2**n,))
