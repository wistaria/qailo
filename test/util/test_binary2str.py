from qailo.util.binary2str import binary2str


def test_binary2str():
    assert binary2str(4, 0b1011) == "1011"
    assert binary2str(6, 0b1110) == "001110"
