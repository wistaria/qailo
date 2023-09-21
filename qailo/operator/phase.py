import numpy as np

def s():
    return np.array([[1, 0],
                     [0, 1j]])

def t():
    return np.array([[1, 0],
                     [0, np.exp(1j*np.pi/4)]])
