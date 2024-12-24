import numpy as np
from utils import dist_spuare
def rbf(x1,x2,l,s):
    return l*np.exp(-s*dist_spuare(x1,x2))

if __name__ == '__main__':
    x = np.linspace(-np.pi, np.pi, 100)
    rbf(x,x,*(1,2))
