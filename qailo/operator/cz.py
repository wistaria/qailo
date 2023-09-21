import numpy as np

def cz():
    return np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]]).reshape([2, 2, 2, 2])
