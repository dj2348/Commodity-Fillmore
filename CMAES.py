import numpy as np
from numpy import exp, sqrt, log, pi
import pandas as pd
import matplotlib.pyplot as plt
import pricer as pr
from calibration import vol_g_fourier
from scipy.interpolate import CubicSpline, interp1d

def loss_function(theta_e_grid, theta_g_grid):

    month_start = np.arange(0, 241)[::20] / 240
    theta_e = interp1d(month_start, theta_e_grid, kind='linear', fill_value='extrapolate')
    theta_g = interp1d(month_start, theta_g_grid, kind='cubic', fill_value='extrapolate')
    model_params = [np.array([theta_e, theta_g]), 0.9, np.array([5.2, 4.4]), (None, vol_g_fourier), 20, True]
    cur_mkt_val = [82, 9.52]
    maturity = 1 + 1/12
    sim_info = [10000, 240]
    model = pr.monte_carlo_simulator(model_params, cur_mkt_val, maturity, sim_info)
    # history = model.rolling_vol_sim()
    history = model.fourier_vol_sim(use_fourier=True)
    fut_e, fut_g = pr.energy_futures(history)
    opt_e, opt_g = pr.energy_euro_call(history, [82, 9.52], pr.yield_curve)

    fut_e_true = np.array([88.15, 98.35, 116, 124.4, 90.5, 86.25,
                           80.65, 86.05, 96, 97.2, 91.75, 80.8])
    fut_g_true = np.array([9.79, 9.88, 9.97, 10.03, 10.05, 10.12,
                           10.4, 10.75, 10.97, 10.95, 10.71, 9.75])
    opt_e_true = np.array([3.54, 5.89, 9.62, 11.88, 8.95, 10.02,
                           9.85, 10.65, 12.56, 13.36, 13.18, 9.32])
    opt_g_true = np.array([0.39, 0.58, 0.73, 0.89, 1.07, 1.26,
                           1.33, 1.43, 1.58, 1.73, 1.77, 1.42])

    mse = (
           np.abs((fut_e - fut_e_true) / fut_e_true) +
           np.abs((fut_g - fut_g_true) / fut_g_true) +
           np.abs((opt_e - opt_e_true) / opt_e_true) +
           np.abs((opt_g - opt_g_true) / opt_g_true)
    ).sum()

    #print(fut_e)
    #print(opt_e)
    #print(fut_g)
    #print(opt_g)
    return mse

theta_e_grid = np.array([82, 88.15, 98.35, 116, 124.4, 90.5, 86.25,
                           80.65, 86.05, 96, 97.2, 91.75, 80.8])
theta_g_grid = np.array([9.52, 9.79, 9.88, 9.97, 10.03, 10.05, 10.12,
                           10.4, 10.75, 10.97, 10.95, 10.71, 9.75])

month_start = np.arange(0, 241)[::20] / 240
discount_factors = np.exp(-month_start * pr.yield_curve(month_start))
mu0 = np.append(theta_e_grid * discount_factors, theta_g_grid * discount_factors)


''' 
mu0 = np.array([96.00537041,  89.73251019,  96.64551107 ,103.19727224 , 94.79647121,
  80.5376809 , 100.47941732,  96.25538964,  97.27607636 ,100.45526759,
 103.42837099,  92.01313663 , 85.93537823,  10.01101213 ,  9.37272288,
  11.0204165 ,  10.65071141,  10.69538862,  12.07781968 , 10.38217413,
  12.64333106,   9.63889084,  13.94927749,   9.74786636,  14.10211372,
   6.38738668]) # 2.50
'''
mu0 = np.array([95.5236947 ,  89.78366771 , 96.12261782, 103.34125784,  94.74710928,
  80.97943651, 100.1284283 ,  94.90594414,  97.82894813 ,100.4985351,
 103.81378475 , 90.95008292,  85.36549408 ,  9.66726308 ,  9.51380563,
  11.13149721 , 10.12741581,  10.66032203 , 12.25803731 , 10.11406903,
  12.66232517 ,  9.71162115,  14.05394954  , 9.74704256,  14.03629003,
   6.46886351]) # 2.37

sig0 = np.eye(26) * np.diag(np.append(np.linspace(4, 4, 13), np.linspace(4, 4, 13) / 8)) / 15 * 0

n_iter = 10000
lam = 200           # num of new points
k = 2              # num of elites
# start algo
elite_losses = []
mu, sig = mu0, sig0
for i in range(n_iter):
    mu_prev, sig_prev = mu, sig
    new_set = np.random.multivariate_normal(mu, sig, lam)
    new_set = np.abs(new_set)
    losses = []

    for params in new_set:
        theta_e_grid = params[:13]
        theta_g_grid = params[13:]
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



