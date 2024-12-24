import numpy as np
def test_1D(x):
    return np.sin(2*np.pi*x) + 0.3 * np.cos(9*np.pi*x) + 0.5 * np.sin(7*np.pi*x)
def dist_spuare(x1,x2):
    return np.sum(np.square(x1), axis=1).reshape(-1, 1) + np.sum(np.square(x2), axis=1) - 2 * np.dot(x1, x2.T)


def train_main():
    # 训练的主循环
    pass

