from copy import deepcopy

import qailo as q


def oracle(v, target):
    n = q.num_qubits(v)
    for k in range(n):
        if q.util.bit(n, target, k) == 0:
            v = q.apply(v, q.op.x(), [k])
    v = q.apply_seq(v, q.op.controlled_seq(q.op.z(), list(range(n))))
    for k in range(n):
        if q.util.bit(n, target, k) == 0:
            v = q.apply(v, q.op.x(), [k])
    return v


def diffusion(v):
    n = q.num_qubits(v)
    for i in range(n):
        v = q.apply(v, q.op.h(), [i])
        v = q.apply(v, q.op.x(), [i])
    v = q.apply_seq(v, q.op.controlled_seq(q.op.z(), list(range(n))))
    for i in range(n):
        v = q.apply(v, q.op.x(), [i])
        v = q.apply(v, q.op.h(), [i])
    return v


def grover(v, target, iter):
    w = deepcopy(v)
    n = q.num_qubits(w)
    for k in range(n):
        w = q.apply(w, q.op.h(), [k])
    for _ in range(iter):
        w = oracle(w, target)
        w = diffusion(w)
    return w


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        n = 4
        target = q.util.str2binary("0101")
    elif len(sys.argv) == 3:
        n = int(sys.argv[1])
        target = q.util.str2binary(sys.argv[2])
    else:
        msg = "len(sys.argv) must be 1 or 3"
        raise ValueError(msg)
    iter = 2 ** (n // 2)
    print("# number of qbits = {}".format(n))
    print("# target state = {}".format(q.util.binary2str(n, target)))
    print("# iterations = {}".format(iter))
    v = q.sv.zero(n)
    prob = q.probability(grover(v, target, iter))
    for i in range(2**n):
        if i == target:
            print("* {} {}".format(q.util.binary2str(n, i), prob[i]))
        else:
            print("  {} {}".format(q.util.binary2str(n, i), prob[i]))
