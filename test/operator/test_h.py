import numpy as np
import qailo as q
from pytest import approx


def test_h():
    print(q.op.h())
    print(q.op.h())
    assert q.op.is_unitary(q.op.h())
    assert q.op.h() == approx((q.op.x() + q.op.z()) / np.sqrt(2))
