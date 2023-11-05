import qailo as q


def main(use_mps=False):
    if use_mps:
        v = q.mps.MPS_C(q.mps.tensor_decomposition(q.sv.state_vector(3)))
    else:
        v = q.sv.state_vector(3)
    print("input:")
    print("state vector:", q.vector(v))
    print("probabitily:", q.probability(v))

    v = q.apply(v, q.op.h(), [0])
    v = q.apply(v, q.op.h(), [2])
    v = q.apply(v, q.op.cx(), [0, 1])
    v = q.apply(v, q.op.cz(), [1, 2])
    v = q.apply(v, q.op.h(), [2])

    print("output:")
    print("state vector:", q.vector(v))
    print("probabitily:", q.probability(v))
    return v


def plot(v):
    import matplotlib.pyplot as plt

    y = q.probability(v)
    x = range(len(y))
    _, ax = plt.subplots()
    ax.bar(x, y, width=0.5, edgecolor="white", linewidth=0.7)
    plt.show()


if __name__ == "__main__":
    import sys

    v = main()
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(v)
