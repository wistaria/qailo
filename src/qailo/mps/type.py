from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt
from typing_extensions import TypeGuard


class mps(ABC):
    tensors: list[npt.NDArray]
    q2t: list[int]
    t2q: list[int]

    @abstractmethod
    def __init__(self, tensors: list[npt.NDArray], nkeep: int | None = None) -> None: ...

    @abstractmethod
    def _num_qubits(self) -> int: ...

    @abstractmethod
    def _norm(self) -> float: ...

    @abstractmethod
    def _state_vector(self) -> npt.NDArray: ...

    @abstractmethod
    def _tensor(self, t: int) -> npt.NDArray: ...

    @abstractmethod
    def _canonicalize(self, p0: int, p1: int | None = None) -> None: ...

    @abstractmethod
    def _is_canonical(self) -> bool: ...

    @abstractmethod
    def _apply_one(self, p: npt.NDArray, s: int) -> None: ...

    @abstractmethod
    def _apply_two(self, p: npt.NDArray, s: int, reverse: bool = False) -> None: ...


def is_canonical(m: mps) -> bool:
    return m._is_canonical()


def is_mps(m: Any) -> TypeGuard[mps]:
    return isinstance(m, mps)


def num_qubits(m: mps) -> int:
    return len(m.q2t)
