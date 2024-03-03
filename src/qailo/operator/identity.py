from __future__ import annotations

import numpy as np
import numpy.typing as npt


def identity(n: int) -> npt.NDArray:
    return np.identity(2**n).reshape((2,) * (2 * n))
