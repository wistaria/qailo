import qailo as q


def main():
    v = q.sv.zero(3)
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

    import numpy as np

    assert q.is_equal(v[0, 0, 0], 1 / np.sqrt(2))
    assert q.is_equal(v[1, 1, 1], 1 / np.sqrt(2))


if __name__ == "__main__":
    main()
