from copy import deepcopy

import numpy as np
import qailo as q


def qpe(n, u, v):
    m = q.num_qubits(u)
    w = deepcopy(v)
    assert q.num_qubits(w) == m + n

    for p in range(n):
        w = q.apply(w, q.op.h(), [p])

    cp = q.op.controlled(u)
    rep = 1
    for p in range(n):
        for _ in range(rep):
            # print(f"apply cu on {p} and {list(range(n,n+m))}")
            w = q.apply(w, cp, [p] + list(range(n, n + m)))
        rep = rep * 2

    w = q.apply_seq(w, q.alg.qft.inverse_qft_seq(n))
    return w


if __name__ == "__main__":
    n = 3
    phi = 2 * np.pi * 0.7
    u = q.op.p(phi)
    v = q.sv.zero()
    v = q.apply(v, q.op.x())
    v = q.sv.product_state([q.sv.zero(n), v])
    v = qpe(n, u, v)
    prob = q.probability(v, list(range(n)))
    for i in range(len(prob)):
        print("{} {}".format(q.util.binary2str(n, i), prob[i]))
