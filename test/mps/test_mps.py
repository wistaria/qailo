import qailo as q
from pytest import approx


def test_mps():
    n = 4
    c = q.util.str2binary("1100")
    mps = q.mps.mps(n, c)
    print(n, c, mps)
    sv = q.mps.state_vector(mps)
    print(sv)
    print(q.sv.state_vector(n, c))
    v = q.sv.vector(sv)
    print(v)

    assert sv == approx(q.sv.state_vector(n, c))


if __name__ == "__main__":
    test_mps()
