from qailo.util.binary2str import binary2str, str2binary


def test_binary2str():
    assert binary2str(4, 0b1011) == "1011"
    assert binary2str(6, 0b1110) == "001110"

    assert str2binary("0000") == 0
    assert str2binary("0010") == 2
    assert str2binary("0110") == 6
    assert str2binary("1000") == 8
