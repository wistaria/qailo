from __future__ import annotations

import numpy as np
import numpy.typing as npt


def is_close(p0: npt.NDArray, p1: npt.NDArray) -> bool:
    return p0.shape == p1.shape and np.allclose(p0, p1)
