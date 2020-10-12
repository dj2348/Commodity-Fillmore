import numpy as np
from numpy import exp, sqrt, log, pi
import pandas as pd
import matplotlib.pyplot as plt
import pricer as pr
from calibration import vol_g_fourier
from scipy.interpolate import CubicSpline, interp1d

def loss_function(theta_e_grid, theta_g_grid):

    month_start = np.arange(0, 241)[::20] / 240
    theta_e = interp1d(month_start, theta_e_grid)
    theta_g = interp1d(month_start, theta_g_grid)
    model_params = [np.array([theta_e, theta_g]), 0.9, np.array([5.2, 4.4]), (None, vol_g_fourier), 20]
    cur_mkt_val = [82, 9.52]
    maturity = 1
    sim_info = [10000, 240]
    model = pr.monte_carlo_simulator(model_params, cur_mkt_val, maturity, sim_info)
    # history = model.rolling_vol_sim()
    history = model.fourier_vol_sim()
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
loss_function(theta_e_grid, theta_g_grid)

month_start = np.arange(0, 241)[::20] / 240
discount_factors = np.exp(-month_start * pr.yield_curve(month_start))
mu0 = np.append(theta_e_grid * discount_factors, theta_g_grid * discount_factors)

mu0 = np.array([92.27899971,  91.50059096,  95.96097533 ,105.74685961, 104.31388194,
  80.91601016 , 96.90298968 , 92.25442685,  96.73748988 ,103.07517094,
 101.83845693 , 93.06977929 , 75.4694364 ,  11.33246507,   9.05474893,
  10.55578234 , 11.03357923 ,  9.98104472 , 12.74295717 ,  9.34372443,
  12.86258062 ,  9.03780083 , 14.57230617 ,  9.74254109  ,14.11576573,
   6.81431898])
np.array([92.36802601 , 90.62305145 , 97.826605 ,  103.23289075 , 98.24075161,
  82.68390815 ,100.31479264 , 90.74603741 , 97.46076898 ,101.57405496,
 104.81820695 , 93.58102432,  79.75864284,  10.59005443,   9.119171,
  10.56497568 , 10.96217962,  10.41287536,  12.6572481 ,   9.9583689,
  12.51036942 ,  9.35891164 , 14.82079736 ,  9.06908409,  14.25137224,
   6.32823412])
sig0 = np.eye(26) * np.diag(np.append(np.linspace(5, 12, 13), np.linspace(5, 12, 13) / 8))

n_iter = 10000
lam = 400           # num of new points
k = 10              # num of elites
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
        if elite_losses[-1] - np.min(elite_losses) > 1 * 10**(-2):      # if error become larger, does not change
            mu = mu_prev
            sig = sig_prev
        elif np.abs(elite_losses[-1] - elite_losses[-2]) < 2 * 10**(-2):    # if little improvement, reset sig
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



