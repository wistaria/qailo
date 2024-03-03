# ref: https://learn.qiskit.org/course/ch-algorithms/quantum-fourier-transform

from __future__ import annotations

import numpy as np

import qailo as q
from qailo.util.helpertype import OPSeqElement


def qft_rotations_seq(n: int) -> list[OPSeqElement]:
    seq: list[OPSeqElement] = []
    if n == 0:
        return seq
    n -= 1
    # print(f"H on [{n}]")
    seq.append(OPSeqElement(q.op.h(), [n]))
    for p in range(n):
        # print(f"CP(pi/{2**(n-p)} on [{p}, {n}]")
        seq.append(OPSeqElement(q.op.cp(np.pi / 2 ** (n - p)), [p, n]))
    seq += qft_rotations_seq(n)
    return seq


def swap_registers_seq(n: int) -> list[OPSeqElement]:
    seq: list[OPSeqElement] = []
    for p in range(n // 2):
        # print(f"swap on [{p}, {n-p-1}]")
        seq.append(OPSeqElement(q.op.swap(), [p, n - p - 1]))
    return seq


def qft_seq(n: int) -> list[OPSeqElement]:
    """QFT on the first n qubits in circuit"""
    return qft_rotations_seq(n) + swap_registers_seq(n)


def inverse_qft_seq(n: int) -> list[OPSeqElement]:
    return q.op.inverse_seq(qft_seq(n))
