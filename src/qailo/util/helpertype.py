from __future__ import annotations

from typing import NamedTuple

import numpy.typing as npt


class OPSeqElement(NamedTuple):
    p: npt.NDArray
    qubit: list[int]
