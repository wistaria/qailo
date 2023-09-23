import qailo as q


def test_replace():
    assert q.util.replace("abcdef", 2, "C") == "abCdef"
    assert q.util.replace("abcdef", -2, "E") == "abcdEf"
