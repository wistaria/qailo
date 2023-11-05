from qailo.util.bitops import bit


def test_bit():
    assert bit(3, 0b100, 0) == 1
    assert bit(3, 0b100, 1) == 0
    assert bit(3, 0b100, 2) == 0
