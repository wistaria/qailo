from __future__ import annotations

from typing import Container, Iterable, overload

import numpy as np
import numpy.typing as npt

from . import mps
from . import operator as op
from . import state_vector as sv
from .mps import type as mpstype
from .util.helpertype import OPSeqElement


@overload
def apply(
    v: npt.NDArray, p: npt.NDArray, pos: list[int] | None = None
) -> npt.NDArray: ...


@overload
def apply(
    v: mpstype.mps, p: npt.NDArray, pos: list[int] | None = None
) -> mpstype.mps: ...


def apply(
    v: npt.NDArray | mpstype.mps, p: npt.NDArray, pos: list[int] | None = None
) -> npt.NDArray | mpstype.mps:
    if sv.is_state_vector(v):
        v = sv.apply(v, p, pos)
    elif mps.is_mps(v):
        v = mps.apply(v, p, pos)
    else:
        assert False
    return v


@overload
def apply_seq(v: npt.NDArray, seq: Iterable[OPSeqElement]) -> npt.NDArray: ...


@overload
def apply_seq(v: mpstype.mps, seq: Iterable[OPSeqElement]) -> mpstype.mps: ...


def apply_seq(
    v: npt.NDArray | mpstype.mps, seq: Iterable[OPSeqElement]
) -> npt.NDArray | mpstype.mps:
    if sv.is_state_vector(v):
        v = sv.apply_seq(v, seq)
    elif mps.is_mps(v):
        v = mps.apply_seq(v, seq)
    else:
        assert False
    return v


def norm(v: npt.NDArray | mpstype.mps) -> float:
    if sv.is_state_vector(v):
        return float(np.linalg.norm(v))
    elif mps.is_mps(v):
        return v._norm()
    else:
        assert False


def num_qubits(v: npt.NDArray | mpstype.mps) -> int:
    if sv.is_state_vector(v):
        return sv.num_qubits(v)
    elif op.is_operator(v):
        return op.num_qubits(v)
    elif mps.is_mps(v):
        return mps.num_qubits(v)
    assert False


def probability(
    v: npt.NDArray | mpstype.mps, pos: Container[int] | None = None
) -> npt.NDArray:
    if sv.is_state_vector(v):
        return sv.probability(v, pos)
    elif mps.is_mps(v):
        return sv.probability(mps.state_vector(v), pos)
    assert False


def vector(v: npt.NDArray | mpstype.mps, c: list | None = None) -> npt.NDArray:
    if sv.is_state_vector(v):
        return sv.vector(v, c)
    elif mps.is_mps(v):
        return sv.vector(v._state_vector(), c)
    assert False
