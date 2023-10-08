def replace(s, i, c):
    if i < 0:
        i += len(s)

    return s[:i] + c + s[i + 1 :]
