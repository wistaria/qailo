from __future__ import annotations

from abc import ABC, abstractmethod

import numpy.typing as npt
from typing_extensions import SupportsIndex


class mps(ABC):
    @abstractmethod
    def __init__(self, tensors: list[npt.NDArray], nkeep: int | None) -> None: ...

    @abstractmethod
    def _num_qubits(self) -> int: ...

    @abstractmethod
    def _norm(self) -> float: ...

    @abstractmethod
    def _state_vector(self) -> npt.NDArray: ...

    @abstractmethod
    def _tensor(self, t: SupportsIndex | slice) -> npt.NDArray: ...

    @abstractmethod
    def _canonicalize(self, p0: int, p1: int | None) -> None: ...

    @abstractmethod
    def _is_canonical(self) -> bool: ...

    @abstractmethod
    def _apply_one(self, p: npt.NDArray, s: int) -> None: ...

    @abstractmethod
    def _apply_two(self, p: npt.NDArray, s: int, reverse: bool | None) -> None: ...


def is_canonical(m):
    return m._is_canonical()


def is_mps(m):
    return isinstance(m, mps)


def num_qubits(m):
    return len(m.q2t)
