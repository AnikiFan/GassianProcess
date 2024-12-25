from models.MyGP import GP,SparseGP
import numpy as np
import matplotlib.pyplot as plt
from utils import test_1D
import time
from models.kernels import rbf
from tqdm import tqdm

NX = 1000
X = np.random.uniform(low=-10, high=10, size=(NX, 1))
X_test = np.linspace(-1, 1, 10000).reshape(-1, 1)

gp = GP(f=test_1D, kernel=rbf, theta=[1, 1], bound=[[1e-5, None], [1e-5, None]])

start_time = time.perf_counter()
gp.fit(X)
end_time = time.perf_counter()
gp_fit_time = end_time - start_time

start_time = time.perf_counter()
gp_m,_ = gp.predict(X_test)
end_time = time.perf_counter()
gp_predict_time = end_time - start_time

gp_mse = np.mean((gp_m-test_1D(X_test))**2)

nzs = [i*10 for i in range(1,11)]
sgp_mse = []
sgp_fit_time = []
sgp_predict_time = []

for nz in tqdm(nzs):
    sgp = SparseGP(f=test_1D, kernel=rbf, theta=[1, 1], bound=[[1e-5, None], [1e-5, None]])

    start_time = time.perf_counter()
    sgp.fit(X, nz)
    end_time = time.perf_counter()
    sgp_fit_time.append(end_time - start_time)

    start_time = time.perf_counter()
    sgp_m,_ = sgp.predict(X_test)
    end_time = time.perf_counter()
    sgp_predict_time.append(end_time - start_time)

    sgp_mse.append(np.mean((sgp_m-test_1D(X_test))**2))

plt.figure(figsize=(16,8))
plt.subplot(131)
plt.title('MSE')
plt.plot(nzs, sgp_mse, label='Sparse GP')
plt.xlabel('#Pseudo Data')
plt.axhline(y=gp_mse,label=f'GP with {NX} samples',linestyle='--',color='r')
plt.legend()

plt.subplot(132)
plt.title('Fit Time')
plt.plot(nzs, sgp_fit_time, label='Sparse GP')
plt.xlabel('#Pseudo Data')
plt.axhline(y=gp_fit_time,label=f'GP with {NX} samples',linestyle='--',color='r')
plt.legend()

plt.subplot(133)
plt.title('Predict Time')
plt.plot(nzs, sgp_predict_time, label='Sparse GP')
plt.xlabel('#Pseudo Data')
plt.axhline(y=gp_predict_time,label=f'GP with {NX} samples',linestyle='--',color='r')
plt.legend()
plt.show()
