import qailo as q

def oracle(v):
    n = len(v.shape)
    return q.sv.multiply(q.op.cz(n), v, range(n))

def diffusion(v):
    n = len(v.shape)
    for i in range(n):
        v = q.sv.multiply(q.op.h(), v, [i])
        v = q.sv.multiply(q.op.x(), v, [i])
    v = q.sv.multiply(q.op.cz(n), v, range(n))
    for i in range(n):
        v = q.sv.multiply(q.op.x(), v, [i])
        v = q.sv.multiply(q.op.h(), v, [i])
    return v

def grover(n, iter):
    v = q.sv.zero(n)
    for i in range(n):
        v = q.sv.multiply(q.op.h(), v, [i])
    for k in range(iter):
        v = oracle(v)
        v = diffusion(v)
    return v
    
if __name__ == '__main__':
    n = 8
    maxiter = 2**n
    print("# number of qbits =", n)
    for iter in range(maxiter):
        prob = q.sv.probability(grover(n, iter))
        print("{} {} {}".format(iter, prob[0], prob[-1]))
