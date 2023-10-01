import qailo as q


def test_binary2str():
    assert q.util.binary2str(4, 0b1011) == "1011"
    assert q.util.binary2str(6, 0b1110) == "001110"
