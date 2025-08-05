from __future__ import annotations

from pytest import approx

import qailo as q
from qailo.util.helpertype import OPSeqElement


def test_apply_seq():
    for n in range(1, 8):
        v0 = q.sv.zero(n)
        v1 = q.sv.zero(n)
        seq: list[OPSeqElement] = []
        for i in range(n):
            v1 = q.apply(v1, q.op.h(), [i])
            seq.append(OPSeqElement(q.op.h(), [i]))
        v2 = q.sv.apply_seq(v0, seq)
        assert q.sv.fidelity(v0, v0) == approx(1)
        assert q.sv.fidelity(v1, v1) == approx(1)
        assert q.sv.fidelity(v2, v2) == approx(1)
        assert q.sv.fidelity(v1, v2) == approx(1)
        assert q.sv.fidelity(v0, v1) == approx(1 / 2**n)
        assert q.sv.fidelity(v0, v2) == approx(1 / 2**n)
