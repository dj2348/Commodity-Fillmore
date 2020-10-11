import numpy as np
import pandas as pd
from scipy.stats import norm

class monte_carlo_simulator:
    def __init__(self, model_params, cur_mkt_val, product_info, sim_info, model: str):
        '''
        model: Fourier or Rolling
        model_params: theta (array of func of t, days. [theta_e, theta_g]), rho, alpha(array of size 2, [alpha_e, alpha_g]), vol, win_len
        cur_mkt_val: e0, g0
        product_info: T, K, payoff (defined as func of history and K)
        sim_info: num_sim, n_steps
        '''
        self._model = model
        self._model_params = model_params
        self._cur_mkt_val = cur_mkt_val
        self._product_info = product_info
        self._sim_info = sim_info
    
    def _fourier_vol_sim(self):
        e0, g0, _ = self._cur_mkt_val
        theta, rho, alpha, vol, _ = self._model_params
        alpha_e, alpha_g = alpha
        theta_e, theta_g = theta
        vol_e, vol_g = vol

        T, _, _ = self._product_info
        num_sim, n_steps = self._sim_info
        delta_t = T/n_steps
        et = np.ones((num_sim, n_steps+1))*e0
        gt = np.ones((num_sim, n_steps+1))*g0
        Z1 = norm.rvs(size = num_sim*n_steps).reshape(num_sim, n_steps)
        Z2 = norm.rvs(size = num_sim*n_steps).reshape(num_sim, n_steps)
        B1 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
       
        for i in range(n_steps):
            et[:,i+1] = et[:,i] + alpha_e * (theta_e(i) - et[:,i])*delta_t + vol_e(i/252)*np.sqrt(delta_t)*Z1[:,i]
            gt[:,i+1] = gt[:,i] - alpha_g * (theta_g(i) - gt[:,i])*delta_t + vol_g(i/252)*np.sqrt(delta_t)*B1[:,i]

        return et, gt
 
    def _rolling_vol_sim(self):
        def rolling_vol(i_day, mat, win_len):
            if i_day + 1 < win_len:
                return np.sqrt(np.sum(np.diff(mat[:,0:i_day], axis = 1)**2, axis = 1)/(i_day + 1)* 250) # Not sure here
            else:
                return np.sqrt(np.sum(np.diff(mat[:,i_day + 1 - win_len:i_day+1], axis = 1)**2, axis = 1)/(i_day + 1)* 250)

        e0, g0, _ = self._cur_mkt_val
        theta, rho, alpha, _, win_len = self._model_params
        alpha_e, alpha_g = alpha
        theta_e, theta_g = theta

        T, _, _ = self._product_info
        num_sim, n_steps = self._sim_info
        delta_t = T/n_steps
        et = np.ones((num_sim, n_steps+1))*e0
        gt = np.ones((num_sim, n_steps+1))*g0
        Z1 = norm.rvs(size = num_sim*n_steps).reshape(num_sim, n_steps)
        Z2 = norm.rvs(size = num_sim*n_steps).reshape(num_sim, n_steps)
        B1 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
       
        for i in range(n_steps):
            et[:,i+1] = et[:,i] + alpha_e * (theta_e(i) - et[:,i])*delta_t + rolling_vol(i, et, win_len)*np.sqrt(delta_t)*Z1[:,i]
            gt[:,i+1] = gt[:,i] - alpha_g * (theta_g(i) - gt[:,i])*delta_t + rolling_vol(i, gt, win_len)*np.sqrt(delta_t)*B1[:,i]

        return et, gt
    
    # def _futures_sim(self):
    #     '''
    #     Simulate the futures price at T: E(S_T | filtration_{T-1}), S_T the futures at month T
    #     '''
    #     if model == 'Fourier':
    #         et, gt = self._fourier_vol_sim()
    #         et = et[:,-20] 
    #         gt = gt[:,-20]

            



def energy_futures(history):
    '''
    Monthly Block Futures Contract：payoff is the aggregate of daily pnl defined by (spot - delivery price)
    '''
    length = history[0].shape[1]
    return np.mean(np.sum(history[0][:, length-20:length], axis = 1)), np.mean(np.sum(history[1][:, length-20:length], axis = 1))




def energy_euro_call(history, K, r, T):
    '''
    T: expiry
    Monthly Block Call Options: underlying is the futures and option expiry = futures delivery.. well it's the fucking spot
    '''
    length = history[0].shape[1]
    et, gt = history

    return np.mean(np.maximum(np.sum(et[:, length-20:length], axis = 1) - K, 0))*np.exp(-r(T)*T), np.mean(np.maximum(np.sum(gt[:, length-20:length], axis = 1) - K, 0))*np.exp(-r(T)*T)

