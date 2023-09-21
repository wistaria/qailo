import math
import numpy as np

def rx(p):
    return np.array([[math.cos(p), -1j*math.sin(p)],
                     [1j*math.sin(p), math.cos(p)]])
