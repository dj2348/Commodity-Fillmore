import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d


class monte_carlo_simulator:
    def __init__(self, model_params, cur_mkt_val, maturity, sim_info):
        '''
        model_params: theta (array of func of t, days. [theta_e, theta_g]), rho, alpha(array of size 2, [alpha_e, alpha_g]), vol, win_len
        cur_mkt_val: e0, g0
        maturity: T
        sim_info: num_sim, n_steps
        '''

        self._model_params = model_params
        self._cur_mkt_val = cur_mkt_val
        self._maturity = maturity
        self._sim_info = sim_info
    
    def fourier_vol_sim(self):
        e0, g0 = self._cur_mkt_val
        theta, rho, alpha, vol, _ = self._model_params
        alpha_e, alpha_g = alpha
        theta_e, theta_g = theta
        vol_e, vol_g = vol

        T = self._maturity
        num_sim, n_steps = self._sim_info
        delta_t = T/n_steps
        et = np.ones((num_sim, n_steps+1))*e0
        gt = np.ones((num_sim, n_steps+1))*g0
        Z1 = norm.rvs(size = num_sim*n_steps).reshape(num_sim, n_steps)
        Z2 = norm.rvs(size = num_sim*n_steps).reshape(num_sim, n_steps)
        B1 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
       
        for i in range(n_steps):
            et[:,i+1] = et[:,i] + alpha_e * (theta_e(i/240) - et[:,i])*delta_t + vol_e(i/240)*np.sqrt(delta_t)*Z1[:,i]
            gt[:,i+1] = gt[:,i] + alpha_g * (theta_g(i/240) - gt[:,i])*delta_t + vol_g(i/240)*np.sqrt(delta_t)*B1[:,i]

        return et, gt
 
    def rolling_vol_sim(self):
        def rolling_vol(i_day, mat, win_len):
            if i_day + 1 < win_len:
                return np.sqrt(np.sum(np.diff(mat[:,0:i_day], axis = 1)**2, axis = 1)/(i_day + 1)* 250) # Not sure here
            else:
                return np.sqrt(np.sum(np.diff(mat[:,i_day + 1 - win_len:i_day+1], axis = 1)**2, axis = 1)/(i_day + 1)* 250)

        e0, g0 = self._cur_mkt_val
        theta, rho, alpha, _, win_len = self._model_params
        alpha_e, alpha_g = alpha
        theta_e, theta_g = theta

        T = self._maturity
        num_sim, n_steps = self._sim_info
        delta_t = T/n_steps
        et = np.ones((num_sim, n_steps+1))*e0
        gt = np.ones((num_sim, n_steps+1))*g0
        Z1 = norm.rvs(size = num_sim*n_steps).reshape(num_sim, n_steps)
        Z2 = norm.rvs(size = num_sim*n_steps).reshape(num_sim, n_steps)
        B1 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
       
        for i in range(n_steps):
            et[:,i+1] = et[:,i] + alpha_e * (theta_e(i/240) - et[:,i])*delta_t + rolling_vol(i, et, win_len)*np.sqrt(delta_t)*Z1[:,i]
            gt[:,i+1] = gt[:,i] + alpha_g * (theta_g(i/240) - gt[:,i])*delta_t + rolling_vol(i, gt, win_len)*np.sqrt(delta_t)*B1[:,i]

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
    Monthly Block Futures Contract：payoff is the expected value of monthly block spot price
    '''
    et, gt = history
    et_lst = np.zeros(12)
    gt_lst = np.zeros(12)
    for i in range(12):
        et_lst[i] = np.mean(np.sum(et[:,  i*20:(i+1)*20+1], axis = 1))
        gt_lst[i] = np.mean(np.sum(gt[:,  i*20:(i+1)*20+1], axis = 1))
    return et_lst, gt_lst




def energy_euro_call(history, K, r, T):
    '''
    T: expiry
    r: yield_curve
    K: [Ke, Kg]
    Monthly Block Call Options: underlying is the futures and option expiry = futures delivery.. well it's the fucking spot
    '''
    et, gt = history
    ke, kg = K
    et_lst = np.zeros(12)
    gt_lst = np.zeros(12)
    for i in range(12):
        tau = ((i+1)*20)/240
        et_lst[i] = np.mean(np.maximum(et[:,(i+1)*20] - ke, 0))*np.exp(-r(tau)*tau)
        gt_lst[i] = np.mean(np.maximum(gt[:,(i+1)*20] - kg, 0))*np.exp(-r(tau)*tau)
    return et_lst, gt_lst

def yield_curve(t):
    '''
    t: annualized maturity: 1mo = 1/12
    '''
    x = np.linspace(0,1,num = 13)
    y = np.array([0, 0.0300, 0.0320, 0.0320, 0.0315, 0.0320, 0.0320, 0.0320, 0.0330, 0.0330, 0.0330, 0.0330, 0.0330])
    return interp1d(x, y, kind='cubic')(np.array([t]))[0]