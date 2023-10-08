import qailo as q


def main():
    v = q.sv.zeros(3)
    print("input:")
    print("state vector:", q.sv.vector(v))
    print("probabitily:", q.sv.probability(v))

    v = q.sv.apply(q.op.h(), v, [0])
    v = q.sv.apply(q.op.h(), v, [2])
    v = q.sv.apply(q.op.cx(), v, [0, 1])
    v = q.sv.apply(q.op.cz(), v, [1, 2])
    v = q.sv.apply(q.op.h(), v, [2])

    print("output:")
    print("state vector:", q.sv.vector(v))
    print("probabitily:", q.sv.probability(v))
    return v


def check(v):
    import numpy as np

    assert q.is_equal(v[0, 0, 0], 1 / np.sqrt(2))
    assert q.is_equal(v[1, 1, 1], 1 / np.sqrt(2))


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
