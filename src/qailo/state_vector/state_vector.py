import numpy as np


def zero():
    return np.array((1, 0)).reshape((2, 1))


def one():
    return np.array((0, 1)).reshape((2, 1))


def product_state(states):
    n = len(states)
    assert n > 0
    v = states[0]
    print(0, v.shape)
    for i in range(1, n):
        v = np.einsum(v.reshape((2,) * (i)), list(range(i)), states[i], [i + 1, i + 2])
        print(i, v.shape)
    return v


def state_vector(n, c=0):
    v = np.zeros(2**n)
    v[c] = 1
    return v.reshape((2,) * n + (1,))
