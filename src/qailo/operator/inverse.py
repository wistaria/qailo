from __future__ import annotations

from typing import Reversible

import numpy.typing as npt

from .hconj import hconj
from .type import is_operator


def inverse_seq(
    seq: Reversible[tuple[npt.NDArray, list[int]]]
) -> list[tuple[npt.NDArray, list[int]]]:
    res: list[tuple[npt.NDArray, list[int]]] = []
    for p, pos in reversed(seq):
        assert is_operator(p)
        res.append((hconj(p), pos))
    return res
