import qailo as q


def oracle(v, target):
    n = q.sv.num_qubits(v)
    for i in range(n):
        if (target >> i) & 1 == 0:
            v = q.sv.apply(q.op.x(), v, [i])
    assert q.op.is_operator(q.op.cz(n))
    assert q.sv.is_state_vector(v)
    v = q.sv.apply(q.op.cz(n), v)
    for i in range(n):
        if (target >> i) & 1 == 0:
            v = q.sv.apply(q.op.x(), v, [i])
    return v


def diffusion(v):
    n = q.sv.num_qubits(v)
    for i in range(n):
        v = q.sv.apply(q.op.h(), v, [i])
        v = q.sv.apply(q.op.x(), v, [i])
    v = q.sv.apply(q.op.cz(n), v)
    for i in range(n):
        v = q.sv.apply(q.op.x(), v, [i])
        v = q.sv.apply(q.op.h(), v, [i])
    return v


def grover(n, target, iter):
    v = q.sv.state_vector(n)
    for i in range(n):
        v = q.sv.apply(q.op.h(), v, [i])
    for k in range(iter):
        v = oracle(v, target)
        v = diffusion(v)
    return v


if __name__ == "__main__":
    n = 4
    target = 0b0110
    iter = 2 ** (n // 2)
    print("# number of qbits = {}".format(n))
    print("# target state = {}".format(q.util.binary2str(n, target)))
    print("# iterations = {}".format(iter))
    prob = q.sv.probability(grover(n, target, iter))
    for i in range(2**n):
        if i == target:
            print("* {} {}".format(q.util.binary2str(n, i), prob[i]))
        else:
            print("  {} {}".format(q.util.binary2str(n, i), prob[i]))
