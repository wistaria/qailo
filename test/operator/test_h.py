import numpy as np
from pytest import approx

import qailo as q


def test_h():
    print(q.op.h())
    print(q.op.h())
    assert q.op.is_unitary(q.op.h())
    assert q.op.h() == approx((q.op.x() + q.op.z()) / np.sqrt(2))
