from qailo.util.bitops import binary2str, str2binary


def test_binary2str():
    assert binary2str(4, 0b1011) == "1011"[::-1]
    assert binary2str(6, 0b1110) == "001110"[::-1]

    assert str2binary("0000") == 0
    assert str2binary("0010") == 4
    assert str2binary("0110") == 6
    assert str2binary("1000") == 1

    n = 6
    for i in range(2**n):
        assert str2binary(binary2str(n, i)) == i
