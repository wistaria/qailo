def replace(s, i, c):
    if i < 0:
        i += len(s)
    assert 0 <= i and i < len(s)
    assert len(c) == 1

    return s[:i] + c + s[i+1:]
