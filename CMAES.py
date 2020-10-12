import numpy as np
from numpy import exp, sqrt, log, pi
import pandas as pd
import matplotlib.pyplot as plt
import pricer_ as pr
from calibration import vol_g_fourier
from scipy.interpolate import CubicSpline, interp1d

mu0 = np.random.randn(13)
sig0 = np.eye(13)

n_iter = 10000
lam = 10           # num of new points
k = 3              # num of elites
# start algo
mu, sig = mu0, sig0

def loss_function(theta_e_grid, theta_g_grid):

    month_start = np.arange(0, 241)[::20]
    theta_e = interp1d(month_start, theta_e_grid)
    theta_g = interp1d(month_start, theta_g_grid)
    model_params = [np.array([theta_e, theta_g]), 0.9, np.array([5.2, 4.4]), (None, vol_g_fourier), 20]
    cur_mkt_val = [82 / 20, 9.52]
    maturity = 1
    sim_info = [10000, 240]
    model = pr.monte_carlo_simulator(model_params, cur_mkt_val, maturity, sim_info)
    # history = model.rolling_vol_sim()
    history = model.fourier_vol_sim()
    fut_e, fut_g = pr.energy_futures(history)
    opt_e, opt_g = pr.energy_euro_call(history, [0.1, 3], pr.yield_curve)

    fut_e_true = np.array([88.15, 98.35, 116, 124.4, 90.5, 86.25,
                           80.65, 86.05, 96, 97.2, 91.75, 80.8])
    fut_g_true = np.array([9.79, 9.88, 9.97, 10.03, 10.05, 10.12,
                           10.4, 10.75, 10.97, 10.95, 10.71, 9.75])
    opt_e_true = np.array([3.54, 5.89, 9.62, 11.88, 8.95, 10.02,
                           9.85, 10.65, 12.56, 13.36, 13.18, 9.32])
    opt_g_true = np.array([0.39, 0.58, 0.73, 0.89, 1.07, 1.26,
                           1.33, 1.43, 1.58, 1.73, 1.77, 1.42])

    mse = (fut_e - fut_e_true)**2 + (fut_g - fut_g_true)**2 + \
          (opt_e - opt_e_true**2) + (opt_g - opt_g_true)**2

    return mse

theta_e_grid = [1, 1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1]
theta_g_grid = [1, 1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1]
loss_function(theta_e_grid, theta_g_grid)

for i in range(n_iter):
    new_set = np.random.multivariate_normal(mu, sig, lam)
    losses = []
    for params in new_set:
        losses += [loss_function(theta_e_grid, theta_g_grid)]
    losses = np.array(losses)
    rank = np.argsort(losses)
    print('Elite losses', losses[rank[:k]].mean())
    elites = new_set[rank[:k]]
    elites = pd.DataFrame(elites)
    mu = elites.mean().to_numpy()
    sig = elites.cov().to_numpy()



