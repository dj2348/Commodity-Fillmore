import numpy as np
from numpy import exp, sqrt, log, pi
import pandas as pd
import matplotlib.pyplot as plt
import pricer as pr
from fourierVol import VolGFourier
from scipy.interpolate import CubicSpline, interp1d

vol_g_fourier = VolGFourier()
def loss_function(theta_e_grid, theta_g_grid):

    month_start = np.arange(0, 261)[::20] / 240
    theta_e = interp1d(month_start, theta_e_grid, kind='linear')
    theta_g = interp1d(month_start, theta_g_grid, kind='linear')
    model_params = [np.array([theta_e, theta_g]), (None, vol_g_fourier), 20, True]
    cur_mkt_val = [82, 9.52]
    maturity = 1 + 1/12
    sim_info = [10000, 260]
    model = pr.monte_carlo_simulator(model_params, cur_mkt_val, maturity, sim_info)
    # history = model.rolling_vol_sim()
    history = model.fourier_vol_sim(use_fourier=True)
    fut_e, fut_g = pr.energy_futures(history)
    opt_e, opt_g = pr.energy_euro_call(history, [82, 9.52], pr.yield_curve)

    fut_e_true = np.array([82, 88.15, 98.35, 116, 124.4, 90.5, 86.25,
                           80.65, 86.05, 96, 97.2, 91.75, 80.8])
    fut_g_true = np.array([9.6, 9.79, 9.88, 9.97, 10.03, 10.05, 10.12,
                           10.4, 10.75, 10.97, 10.95, 10.71, 9.75])
    opt_e_true = np.array([3.54, 5.89, 9.62, 11.88, 8.95, 10.02,
                           9.85, 10.65, 12.56, 13.36, 13.18, 9.32])
    opt_g_true = np.array([0.39, 0.58, 0.73, 0.89, 1.07, 1.26,
                           1.33, 1.43, 1.58, 1.73, 1.77, 1.42])

    mae = np.abs((fut_e - fut_e_true) / fut_e_true).sum() +\
           np.abs((fut_g - fut_g_true) / fut_g_true).sum() +\
           np.abs((opt_e - opt_e_true) / opt_e_true).sum() +\
           np.abs((opt_g - opt_g_true) / opt_g_true).sum()

    mse = (((fut_e - fut_e_true) / fut_e_true)**2).sum() +\
          (((fut_g - fut_g_true) / fut_g_true)**2).sum() +\
          (((opt_e - opt_e_true) / opt_e_true)**2).sum() +\
          (((opt_g - opt_g_true) / opt_g_true)**2).sum()
    #print(fut_e)
    #print(opt_e)
    #print(fut_g)
    #print(opt_g)
    return mae

theta_e_grid = np.array([82, 88.15, 98.35, 116, 124.4, 90.5, 86.25,
                           80.65, 86.05, 96, 97.2, 91.75, 80.8, 80.8])
theta_g_grid = np.array([9.52, 9.79, 9.88, 9.97, 10.03, 10.05, 10.12,
                           10.4, 10.75, 10.97, 10.95, 10.71, 9.75, 9.75])
loss_function(theta_e_grid, theta_g_grid)


month_start = np.arange(0, 261)[::20] / 240
discount_factors = np.exp(-month_start * pr.yield_curve(month_start))
mu0 = np.append(theta_e_grid * discount_factors, theta_g_grid * discount_factors)

mu0 = np.array([95.77624878,  89.33899388,  94.97090706, 103.38369855,  94.94380183,
  81.75487088, 100.25696859,  93.62995212,  98.17644888,  99.3983192,
 102.46467819,  91.16151008,  84.44134567,  79.31319372 ,  9.43545258,
   9.72969004 , 10.75273553,  10.38411311,  10.85842813 , 11.78349431,
  10.25351524 , 12.55570602,   9.92906091 , 14.01353142 ,  9.30671949,
  14.28954042 ,  6.77069735 ,  6.87730178])

mu0 = np.array([96.03099537,  88.96756393,  95.15456025, 103.43338149,  94.66244055,
  81.28469814,  99.4890207 ,  94.22996031,  98.77832987 , 98.45671238,
 103.33602985,  91.78342438,  83.64211702 , 79.62639204 ,  9.52345856,
   9.74838331,  10.78937404 , 10.51459812,  10.83462434, 11.68384713,
  10.1475244 ,  12.73082681,  10.1454398,   13.61106297 ,  9.52353396,
  14.07108164 ,  6.91007521,   6.64479901])

sig0 = np.eye(28) * np.diag(np.append(np.linspace(4, 4, 14), np.linspace(4, 4, 14) / 8)) / 5
sig0 = np.eye(28) * np.diag(np.append(
                                np.array([0, 0, 0, 4, 4, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0]),
                                np.linspace(4, 4, 14) / 8
                                )) / 5

n_iter = 10000
lam = 200           # num of new points
k = 10             # num of elites
# start algo
elite_losses = []
mu, sig = mu0, sig0
for i in range(n_iter):
    mu_prev, sig_prev = mu, sig
    new_set = np.random.multivariate_normal(mu, sig, lam)
    new_set = np.abs(new_set)
    losses = []

    for params in new_set:
        theta_e_grid = params[:14]
        theta_g_grid = params[14:]
        losses += [loss_function(theta_e_grid, theta_g_grid)]
    losses = np.array(losses)
    rank = np.argsort(losses)
    elite_losses.append(losses[rank[:k]].mean())

    print(i, 'Elite losses', elite_losses[-1])
    with open("caliblog.txt", 'a') as file1:
        file1.write(str(i) + ' Elite losses ' + str(elite_losses[-1]) + '\n')

    elites = new_set[rank[:k]]
    elites = pd.DataFrame(elites)
    mu = elites.mean().to_numpy()
    sig = elites.cov().to_numpy()
    if i == 0:
        with open("caliblog.txt", 'a') as file1:
            file1.write(' '.join([str(_) for _ in mu]))
            file1.write('\n')
    if i >= 1:
        if elite_losses[-1] - np.min(elite_losses) > 1 * 10**(-3):      # if error become larger, does not change
            mu = mu_prev
            sig = sig_prev
        elif np.abs(elite_losses[-1] - elite_losses[-2]) < 1 * 10**(-3):    # if little improvement, reset sig
            sig0 = sig0 / 2
            sig = sig0
            print('Reset covariance')
            print(mu)
            with open("caliblog.txt", 'a') as file1:
                file1.write('Reset covariance')
                file1.write(' '.join([str(_) for _ in mu]))
                file1.write('\n')
        else:
            print(mu)
            with open("caliblog.txt", 'a') as file1:
                file1.write(' '.join([str(_) for _ in mu]))
                file1.write('\n')



