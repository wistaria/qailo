import qailo as q
from qailo.mps_t.tpool import tpool


def test_mps_t():
    v = q.mps_t.zero(3)

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

    print("# tensor pool")
    for id, tp in enumerate(v.tp.tpool):
        print(f"{id} {tp[0].shape} {tp[1]} {tp[2]}")
    assert len(v.tp.tpool) == 25

    print("# generator pool")
    for id, gp in enumerate(v.tp.gpool):
        print(f"{id} {gp[0].shape} {gp[1]} {gp[2].shape} {gp[3].shape}")
    assert len(v.tp.gpool) == 2

    prefix = "test_mps_t"
    v.tp._dump(prefix)

    tp = tpool()
    tp._load(prefix)


if __name__ == "__main__":
    test_mps_t()
