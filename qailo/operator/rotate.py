import numpy as np

def rx(p):
    return np.array([[np.cos(p/2), -1j*np.sin(p/2)],
                     [-1j*np.sin(p/2), np.cos(p/2)]])

def ry(p):
    return np.array([[np.cos(p/2), -np.sin(p/2)],
                     [np.sin(p/2), np.cos(p/2)]])

def rz(p):
    return np.array([[np.exp(-1j*p/2), 0],
                     [0, np.exp(1j*p/2)]])
