import numpy as np

import qailo as q


def test_h():
    print(q.op.h())
    print(q.op.x())
    print(q.op.z())
    assert q.op.is_unitary(q.op.h())
    assert q.op.is_close(q.op.h(), (q.op.x() + q.op.z()) / np.sqrt(2))
