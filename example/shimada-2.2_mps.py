import numpy as np
import qailo as q


def main():
    d = 2
    m = q.mps.MPS_P(q.mps.product_state(3))
    v = q.mps.state_vector(m)
    print("input:")
    print("state vector:", q.sv.vector(v))
    print("probabitily:", q.sv.probability(v))

    m = q.mps.apply(m, q.op.h(), [0], maxdim=d)
    m = q.mps.apply(m, q.op.h(), [2], maxdim=d)
    m = q.mps.apply(m, q.op.cx(), [0, 1], maxdim=d)
    m = q.mps.apply(m, q.op.cz(), [1, 2], maxdim=d)
    m = q.mps.apply(m, q.op.h(), [2], maxdim=d)

    print("output:")
    v = q.mps.state_vector(m)
    print("state vector:", q.sv.vector(v))
    print("probabitily:", q.sv.probability(v))
    return v


def check(v):
    assert np.allclose(v[0, 0, 0], 1 / np.sqrt(2))
    assert np.allclose(v[1, 1, 1], 1 / np.sqrt(2))


def plot(v):
    import matplotlib.pyplot as plt

    y = q.sv.probability(v)
    x = range(len(y))
    _, ax = plt.subplots()
    ax.bar(x, y, width=0.5, edgecolor="white", linewidth=0.7)
    plt.show()


if __name__ == "__main__":
    import sys

    v = main()
    check(v)
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(v)
