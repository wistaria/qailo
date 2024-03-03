from .hconj import hconj
from .type import is_operator


def inverse_seq(seq):
    res = []
    for p, pos in reversed(seq):
        assert is_operator(p)
        res.append([hconj(p), pos])
    return res
