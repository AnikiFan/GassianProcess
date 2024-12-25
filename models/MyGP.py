import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from models.kernels import *
from utils.utils import *
import seaborn as sns


class GP:
    def __init__(self, kernel, theta, bound, f=None,noise=1e-5,nugget=1e-10):
        """

        :param f: y = f(X)
        :param kernel: kernel(X,Y,theta)，其中theta为超参数列表
        :param theta: 超参数初始值
        :param bound: 超参数上下界
        :param noise: 噪声
        """
        self.f = f
        self.kernel = kernel
        self.theta = theta
        self.noise = noise
        self.bound = bound
        self.nugget = nugget
        self._X = None
        self._y = None
        self._m = None
        self._s = None
        self._X_test = None
        self._y_test = None

    def fit(self, X,y=None):
        assert y is not None or self.f is not None
        self._X = X
        if y is not None:
            self._y = y
        else:
            self._y = self.f(X)
        
        if isinstance(X, pd.DataFrame):
            self._X = X.values
        if isinstance(y, pd.Series):
            self._y = y.values

        def marginal_likelihood(hyperparameters):
            theta = hyperparameters[:-1]
            noise = hyperparameters[-1]
            K = self.kernel(self._X, self._X, *theta) + noise * np.eye(self._X.shape[0])
            return np.sum(np.log(np.diagonal(np.linalg.cholesky(K)))) + \
                0.5 * self._y.T @ np.linalg.inv(K) @ self._y + \
                0.5 * self._X.shape[0] * np.log(2 * np.pi)

        x0 = self.theta
        x0.append(self.noise)
        bounds = self.bound
        bounds.append([1e-5,None])
        res = minimize(marginal_likelihood, x0=x0, bounds=bounds, method='L-BFGS-B')
        self.theta = res.x[:-1]
        self.noise = res.x[-1]

    def predict(self, X_test,y_test=None):
        self._X_test = X_test
        if y_test is not None:
            self._y_test = y_test
        elif self.f is not None:
            self._y_test = self.f(X_test)

        C_N = self.kernel(self._X, self._X, *self.theta) + self.noise * np.eye(self._X.shape[0])
        k = self.kernel(self._X, self._X_test, *self.theta)
        c = self.kernel(self._X_test, self._X_test, *self.theta) + self.nugget * np.eye(self._X_test.shape[0])
        C_N_inv = np.linalg.inv(C_N)
        m = k.T @ C_N_inv @ self._y
        cov = np.clip(c - k.T @ C_N_inv @ k,1e-15,np.inf)
        s = np.sqrt(np.diag(cov) + self.noise)
        self._m = m
        self._s = s
        return m, s

    def plot(self):
        plt.plot(self._X_test.flatten(), self._m.flatten(), label=r'$\mu$')
        if self._y_test is not None:
            plt.plot(self._X_test.flatten(), self._y_test.flatten(), label=r'$f(x)$')
        if self.f is None:
            plt.plot(self._X.flatten(),self._y.flatten(), label='training data',linestyle='dashed')
        else:
            plt.scatter(self._X.flatten(),self._y.flatten(), label='training data',marker='x',color='black')
        plt.fill_between(self._X_test.flatten(), self._m.flatten() - 1.96 * self._s.flatten(),
                         self._m.flatten() + 1.96 * self._s.flatten(), alpha=0.2, label=r'$\mu\pm 2\sigma$')
        plt.legend()


class SparseGP:
    def __init__(self, kernel, theta, bound,f=None, noise=1e-5, nugget=1e-10):
        """
        :param y: 训练集标签
        :param f: y = f(X)
        :param kernel: kernel(X,Y,*theta)，其中theta为超参数列表
        :param theta: 超参数初始值
        :param bound: 超参数上下界
        :param noise: 噪声
        :param nugget: 用于保持数值稳定性
        """
        self.f = f
        self.kernel = kernel
        self.theta = theta  
        self.noise = noise
        self.bound = bound
        self.nugget = nugget
        self._X = None
        self._Z = None
        self._m = None
        self._s = None
        self._X_test = None
        self._y_test = None

    def fit(self, X, nz,y=None):
        """
        :param X: 测试集
        :param nz: pseudo data的个数 
        :param y: 测试集标签，如果之前已经提供了f则可以不提供 
        :return: 
        """
        assert y is not None or self.f is not None
        self._X = X
        kmeans = KMeans(n_clusters=nz)
        kmeans.fit(X)
        self._Z = kmeans.cluster_centers_
        if y is not None:
            self._y = y
        else:
            self._y = self.f(X)
        
        if isinstance(X, pd.DataFrame):
            self._X = X.values
        if isinstance(y, pd.Series):
            self._y = y.values


        def marginal_likelihood(hyperparameters):
            theta = hyperparameters[:-1]
            noise = hyperparameters[-1]
            Knn = np.concatenate([self.kernel(self._X[[i],:],self._X[[i],:],*theta) for i in range(self._X.shape[0])]).T
            Kmm = self.kernel(self._Z,self._Z,*theta) + self.nugget * np.eye(self._Z.shape[0])
            Kmn = self.kernel(self._Z,self._X,*theta)
            U = np.linalg.cholesky(Kmm)
            U_inv = np.linalg.inv(U)
            V = U_inv @ Kmn
            D = Knn - np.sum(np.square(V),0) + noise
            D_inv = np.eye(self._X.shape[0])/D
            L = np.linalg.cholesky(np.eye(nz)+V@D_inv@V.T)
            log_determinant =  np.sum(np.log(D)) + 2*np.sum(np.log(L.diagonal()))
            L_inv = np.linalg.inv(L)
            LVD = L_inv@V@D_inv
            K_inv = D_inv - LVD.T@LVD
            return log_determinant + 0.5 * self._y.T @ K_inv @ self._y + 0.5 * self._X.shape[0] * np.log(2 * np.pi)
        x0 = self.theta
        x0.append(self.noise)
        bounds = self.bound
        bounds.append([1e-5,None])
        res = minimize(marginal_likelihood, x0=x0, bounds=bounds, method='L-BFGS-B')
        self.theta = res.x[:-1]
        self.noise = res.x[-1]

    def predict(self, X_test,y_test=None):
        self._X_test = X_test
        if y_test is not None:
            self._y_test = y_test
        elif self.f is not None:
            self._y_test = self.f(X_test)
        Knn = np.concatenate([self.kernel(self._X[[i], :], self._X[[i], :], *self.theta) for i in range(self._X.shape[0])]).T
        Kmm = self.kernel(self._Z, self._Z, *self.theta) + self.nugget * np.eye(self._Z.shape[0])
        Kmn = self.kernel(self._Z, self._X, *self.theta)
        U = np.linalg.cholesky(Kmm)
        U_inv = np.linalg.inv(U)
        V = U_inv @ Kmn
        D = Knn - np.sum(np.square(V), 0) + self.noise
        D_inv = np.eye(self._X.shape[0])/D
        L = np.linalg.cholesky(np.eye(self._Z.shape[0]) + V @ D_inv @ V.T )
        L_inv = np.linalg.inv(L)
        LVD = L_inv@V@D_inv
        K_inv = D_inv - LVD.T@LVD
        Q = self.kernel(X_test,self._Z, *self.theta)@Kmm@Kmn
        K_test = self.kernel(X_test,X_test, *self.theta)
        m = Q @ K_inv @ self._y
        cov = np.clip(K_test - Q @ K_inv @ Q.T,1e-15,np.inf)
        s = np.sqrt(np.diag(cov)+self.noise)
        self._m = m
        self._s = s
        return m, s

    def plot(self):
        plt.plot(self._X_test.flatten(), self._m.flatten(), label=r'$\mu$')
        if self._y_test is not None:
            plt.plot(self._X_test.flatten(), self._y_test.flatten(), label=r'$f(x)$')
        if self.f is None:
            plt.plot(self._X.flatten(),self._y.flatten(), label='training data',linestyle='dashed')
        else:
            plt.scatter(self._X.flatten(),self._y.flatten(), label='training data',marker='x',color='black')
        plt.fill_between(self._X_test.flatten(), self._m.flatten() - 1.96 * self._s.flatten(),
                         self._m.flatten() + 1.96 * self._s.flatten(), alpha=0.2, label=r'$\mu\pm 2\sigma$')
        sns.rugplot(self._Z.flatten(), label='pseudo data',lw=1)
        plt.legend()


if __name__ == '__main__':
    X = np.random.uniform(low=-1, high=1, size=(20, 1))
    X_test = np.linspace(-1, 1, 100).reshape(-1, 1)

    plt.figure(figsize=(16, 8))
    gp = GP(f=test_1D, kernel=rbf, theta=[1, 1], bound=[[1e-5, None], [1e-5, None]])
    gp.fit(X)
    gp.predict(X_test)
    gp.plot()
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(os.curdir,'fig','GP.png'),dpi=600,transparent=True)

    plt.figure(figsize=(16, 8))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        nz = 3*(i+1)
        sgp = SparseGP(f = test_1D, kernel = rbf, theta = [1,1], bound = [[1e-5,None],[1e-5,None]])
        sgp.fit(X, nz)
        sgp.predict(X_test)
        sgp.plot()
        plt.title(f'#pseudo data = {nz}')
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(os.curdir,'fig','SGP.png'),dpi=600,transparent=True)


