import qailo as q


def test_shape():
    assert q.util.shape(3) == (2, 2, 2)
