import qailo as q


def oracle(v, target):
    n = q.num_qubits(v)
    for i in range(n):
        if (target >> (n - i - 1)) & 1 == 0:
            v = q.apply(v, q.op.x(), [i])
    v = q.apply_seq(v, q.op.controlled_seq(q.op.z(), list(range(n))))
    for i in range(n):
        if (target >> (n - i - 1)) & 1 == 0:
            v = q.apply(v, q.op.x(), [i])
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


def grover(n, target, iter, use_mps):
    if use_mps:
        v = q.mps.zero(n)
    else:
        v = q.sv.zero(n)
    for i in range(n):
        v = q.apply(v, q.op.h(), [i])
    for k in range(iter):
        v = oracle(v, target)
        v = diffusion(v)
    return v


if __name__ == "__main__":
    n = 4
    target = q.util.str2binary("0111")
    iter = 2 ** (n // 2)
    print("# number of qbits = {}".format(n))
    print("# target state = {}".format(q.util.binary2str(n, target)))
    print("# iterations = {}".format(iter))
    prob = q.probability(grover(n, target, iter, False))
    for i in range(2**n):
        if i == target:
            print("* {} {}".format(q.util.binary2str(n, i), prob[i]))
        else:
            print("  {} {}".format(q.util.binary2str(n, i), prob[i]))
