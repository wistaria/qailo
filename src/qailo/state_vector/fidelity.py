from __future__ import annotations

import numpy as np
import numpy.typing as npt


def fidelity(v0: npt.NDArray, v1: npt.NDArray) -> float:
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    ret = np.abs(np.vdot(v0, v1)) ** 2
    assert isinstance(ret, float)
    return ret
