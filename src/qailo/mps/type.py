import numpy as np

from .mps import MPS

def is_mps(mps):
    return isinstance(mps, MPS)


def num_qubits(mps):
    return mps.num_qubits()
