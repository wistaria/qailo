from __future__ import annotations


def bit(n: int, s: int, k: int) -> int:
    """extract k-th bit from integer s"""
    return (s >> (n - k - 1)) & 1


def binary2str(n: int, i: int) -> str:
    return bin(i)[2:].zfill(n)[::-1]


def str2binary(s: str) -> int:
    n = len(s)
    c: int = 0
    for i in range(n):
        assert s[i] == "0" or s[i] == "1"
        if s[i] == "1":
            c += 2**i
    return c
