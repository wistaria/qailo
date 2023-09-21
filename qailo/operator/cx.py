import numpy as np

def cx():
    return np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]]).reshape([2, 2, 2, 2])
