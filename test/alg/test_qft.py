import numpy as np
import qailo as q
from pytest import approx


def test_qft():
    n = 3
    target = 5

    # generate result of qft of target
    v = q.sv.state_vector(n)
    for p in range(n):
        v = q.apply(v, q.op.h(), [p])
    v = q.apply(v, q.op.p(target * np.pi / 4), [0])
    v = q.apply(v, q.op.p(target * np.pi / 2), [1])
    v = q.apply(v, q.op.p(target * np.pi), [2])

    # apply inverse qft
    v = q.apply_seq(v, q.alg.inverse_qft_seq(n))

    v = q.vector(v)
    for i in range(2**n):
        if i == target:
            print("* {} {}".format(q.util.binary2str(n, i), v[i]))
            assert v[i] == approx(1)
        else:
            print("  {} {}".format(q.util.binary2str(n, i), v[i]))
            assert v[i] == approx(0)


if __name__ == "__main__":
    test_qft()
