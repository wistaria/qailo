import qailo as q


def oracle(m, target, d):
    n = q.mps.num_qubits(m)
    for i in range(n):
        if (target >> i) & 1 == 0:
            m = q.mps.apply(m, q.op.x(), [i], maxdim=d)
    m = q.mps.apply(m, q.op.control_begin(), [0, 1], maxdim=d)
    for i in range(1, n - 2):
        m = q.mps.apply(m, q.op.control_propagate(), [i, i + 1], maxdim=d)
    m = q.mps.apply(m, q.op.control_end(q.op.z()), [n - 2, n - 1], maxdim=d)
    for i in range(n):
        if (target >> i) & 1 == 0:
            m = q.mps.apply(m, q.op.x(), [i])
    return m


def diffusion(m, d):
    n = q.mps.num_qubits(m)
    for i in range(n):
        m = q.mps.apply(m, q.op.h(), [i], maxdim=d)
        m = q.mps.apply(m, q.op.x(), [i], maxdim=d)
    m = q.mps.apply(m, q.op.control_begin(), [0, 1], maxdim=d)
    for i in range(1, n - 2):
        m = q.mps.apply(m, q.op.control_propagate(), [i, i + 1], maxdim=d)
    m = q.mps.apply(m, q.op.control_end(q.op.z()), [n - 2, n - 1], maxdim=d)
    for i in range(n):
        m = q.mps.apply(m, q.op.x(), [i], maxdim=d)
        m = q.mps.apply(m, q.op.h(), [i], maxdim=d)
    return m


def grover(n, target, iter, d):
    m = q.mps.MPS_P(q.mps.product_state(n))
    for i in range(n):
        m = q.mps.apply(m, q.op.h(), [i], maxdim=d)
    for _ in range(iter):
        m = oracle(m, target, d)
        m = diffusion(m, d)
    return m


if __name__ == "__main__":
    n = 4
    target = 0b0110
    iter = 2 ** (n // 2)
    d = 2
    print("# number of qbits = {}".format(n))
    print("# target state = {}".format(q.util.binary2str(n, target)))
    print("# iterations = {}".format(iter))
    print("# bond dimension = {}".format(d))
    m = grover(n, target, iter, d)
    prob = q.sv.probability(q.mps.state_vector(m))
    for i in range(2**n):
        if i == target:
            print("* {} {}".format(q.util.binary2str(n, i), prob[i]))
        else:
            print("  {} {}".format(q.util.binary2str(n, i), prob[i]))
