import math
import numpy as np

def rz(p):
    return np.array([[math.exp(p*1j), 0],
                     [0, math.exp(-p*1j)]])
