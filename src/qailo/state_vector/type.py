from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeGuard


def is_state_vector(v: Any) -> TypeGuard[np.ndarray]:
    if isinstance(v, np.ndarray):
        return v.shape[-1] == 1 and v.shape[-2] > 1
    else:
        return False


def num_qubits(v: npt.NDArray) -> int:
    return v.ndim - 1
