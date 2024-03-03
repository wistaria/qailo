"""Wrap np.einsum for better type checking."""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt


@overload
def einsum(ss: str, *op: npt.NDArray) -> npt.ArrayLike: ...


@overload
def einsum(op1: np.ndarray, ss1: list[int]) -> npt.ArrayLike: ...


@overload
def einsum(op1: np.ndarray, ss1: list[int], ss_out: list[int]) -> npt.ArrayLike: ...


@overload
def einsum(
    op1: np.ndarray,
    ss1: list[int],
    op2: np.ndarray,
    ss2: list[int],
) -> npt.ArrayLike: ...


@overload
def einsum(
    op1: np.ndarray,
    ss1: list[int],
    op2: np.ndarray,
    ss2: list[int],
    ss_out: list[int],
) -> npt.ArrayLike: ...


def einsum(*args, **kwargs):
    return np.einsum(*args, **kwargs)


@overload
def einsum_cast(ss: str, *op: npt.NDArray) -> npt.NDArray: ...


@overload
def einsum_cast(op1: np.ndarray, ss1: list[int]) -> npt.NDArray: ...


@overload
def einsum_cast(op1: np.ndarray, ss1: list[int], ss_out: list[int]) -> npt.NDArray: ...


@overload
def einsum_cast(
    op1: np.ndarray,
    ss1: list[int],
    op2: np.ndarray,
    ss2: list[int],
) -> npt.NDArray: ...


@overload
def einsum_cast(
    op1: np.ndarray,
    ss1: list[int],
    op2: np.ndarray,
    ss2: list[int],
    ss_out: list[int],
) -> npt.NDArray: ...


def einsum_cast(*args, **kwargs) -> npt.NDArray:
    return np.asarray(np.einsum(*args, **kwargs))
