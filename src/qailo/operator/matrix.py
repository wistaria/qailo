from __future__ import annotations

import typing
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .type import is_density_matrix, is_operator, num_qubits

T_co = TypeVar("T_co", covariant=True, bound=np.generic)


def matrix(op: npt.NDArray[T_co]) -> npt.NDArray[T_co]:
    assert is_density_matrix(op) or is_operator(op)
    
    n = num_qubits(op)
    return op.reshape([2**n, 2**n])
