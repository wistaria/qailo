def bit(n, s, k):
    """extract k-th bit from integer s"""
    return (s >> (n - k - 1)) & 1


def binary2str(n, i):
    return bin(i)[2:].zfill(n)[::-1]


def str2binary(s):
    n = len(s)
    c = 0
    for i in range(n):
        assert s[i] == "0" or s[i] == "1"
        if s[i] == "1":
            c += 2**i
    return c
