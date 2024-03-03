from __future__ import annotations

from typing import Reversible

from .hconj import hconj
from .type import OPAutomaton, is_operator


def inverse_seq(seq: Reversible[OPAutomaton]) -> list[OPAutomaton]:
    res: list[OPAutomaton] = []
    for p, pos in reversed(seq):
        assert is_operator(p)
        res.append(OPAutomaton(hconj(p), pos))
    return res
