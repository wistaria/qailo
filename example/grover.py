import qailo as q


def oracle(v):
    n = len(v.shape)
    return q.sv.apply(q.op.cz(n), v)


def diffusion(v):
    n = len(v.shape)
    for i in range(n):
        v = q.sv.apply(q.op.h(), v, [i])
        v = q.sv.apply(q.op.x(), v, [i])
    v = q.sv.apply(q.op.cz(n), v)
    for i in range(n):
        v = q.sv.apply(q.op.x(), v, [i])
        v = q.sv.apply(q.op.h(), v, [i])
    return v


def grover(n, iter):
    v = q.sv.zero(n)
    for i in range(n):
        v = q.sv.apply(q.op.h(), v, [i])
    for k in range(iter):
        v = oracle(v)
        v = diffusion(v)
    return v


if __name__ == "__main__":
    n = 8
    iter = 2 ** (n // 2)
    print("# number of qbits =", n)
    print("# iterations = ", iter)
    prob = q.sv.probability(grover(n, iter))
    print(prob[0], prob[-1])
