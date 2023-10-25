from qailo.util.strops import replace


def test_replace():
    assert replace("abcdef", 2, "C") == "abCdef"
    assert replace("abcdef", -2, "E") == "abcdEf"
