from __future__ import annotations

from typing import Reversible

from ..util.helpertype import OPSeqElement
from .hconj import hconj
from .type import is_operator


def inverse_seq(seq: Reversible[OPSeqElement]) -> list[OPSeqElement]:
    res: list[OPSeqElement] = []
    for p, pos in reversed(seq):
        assert is_operator(p)
        res.append(OPSeqElement(hconj(p), pos))
    return res
