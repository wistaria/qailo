import numpy as np
import qailo as q


def qpe(n, u, ev):
    m = q.num_qubits(u)
    assert q.num_qubits(ev) == m
    v = q.sv.product_state([q.sv.zero(n), ev])
    assert q.num_qubits(v) == n + m

    for p in range(n):
        v = q.apply(v, q.op.h(), [p])

    cp = q.op.controlled(u)
    rep = 1
    for p in range(n):
        for _ in range(rep):
            # print(f"apply cu on {p} and {list(range(n,n+m))}")
            v = q.apply(v, cp, [p] + list(range(n, n + m)))
        rep = rep * 2

    v = q.apply_seq(v, q.alg.qft.inverse_qft_seq(n))
    return v


if __name__ == "__main__":
    n = 3
    phi = 2 * np.pi * (1 / 3)
    u = q.op.p(phi)
    ev = q.sv.zero()
    ev = q.apply(ev, q.op.x())
    v = qpe(n, u, ev)
    prob = np.diag(q.op.matrix(q.op.trace(q.sv.pure_state(v), [n])).real)
    for i in range(len(prob)):
        print("{} {}".format(q.util.binary2str(n, i), prob[i]))
