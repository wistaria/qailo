from __future__ import annotations

import numpy as np
import numpy.typing as npt


def swap() -> npt.NDArray:
    op = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return op.reshape([2, 2, 2, 2])
