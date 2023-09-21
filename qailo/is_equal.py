import numpy as np

def is_equal(s, t):
    return (np.linalg.norm(s - t)) < 1e-15
