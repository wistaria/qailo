# ref: https://learn.qiskit.org/course/ch-algorithms/quantum-fourier-transform

import numpy as np

import qailo as q


def qft_rotations_seq(n):
    seq = []
    if n == 0:
        return seq
    n -= 1
    # print(f"H on [{n}]")
    seq.append([q.op.h(), [n]])
    for p in range(n):
        # print(f"CP(pi/{2**(n-p)} on [{p}, {n}]")
        seq.append([q.op.cp(np.pi / 2 ** (n - p)), [p, n]])
    seq += qft_rotations_seq(n)
    return seq


def swap_registers_seq(n):
    seq = []
    for p in range(n // 2):
        # print(f"swap on [{p}, {n-p-1}]")
        seq.append([q.op.swap(), [p, n - p - 1]])
    return seq


def qft_seq(n):
    """QFT on the first n qubits in circuit"""
    return qft_rotations_seq(n) + swap_registers_seq(n)


def inverse_qft_seq(n):
    return q.op.inverse_seq(qft_seq(n))
