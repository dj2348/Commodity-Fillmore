import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numpy import exp, sqrt, log, pi
from numpy.fft import fft



class VolGFourier:
    def __init__(self):
        lam_gas = 4.4
        span = 120
        delta_t = 1 / 252

        gas_hisspot = pd.read_excel('gas spot.xlsx')
        gas_hisspot.columns = ['Day', 'Spot']
        gas_hisspot['Day'] = pd.to_datetime(gas_hisspot['Day'])
        gas_hisspot = gas_hisspot.set_index('Day').sort_index()

        gas_hisspot_exp = gas_hisspot.copy()
        gas_hisspot_exp['t'] = np.arange(gas_hisspot.shape[0]) / 252
        gas_hisspot_exp['Spot'] = gas_hisspot_exp['Spot'] * exp(lam_gas * gas_hisspot_exp['t'])
        gas_hisspot_exp['diff'] = gas_hisspot_exp['Spot'] - gas_hisspot_exp['Spot'].shift(1)

        gas_hisret = gas_hisspot - gas_hisspot.shift(1)
        gas_hisret.columns = ['ret']
        gas_hisret['integral'] = (gas_hisspot + gas_hisspot.shift(1)) * delta_t / 2

        gas_hisvol = pd.DataFrame(index=gas_hisret.index, columns=['Vol'])
        for i in range(span, gas_hisret.shape[0]):
            tmp = (gas_hisret.iloc[i - span:i, 0] + lam_gas * gas_hisret.iloc[i - span:i, 1]) ** 2
            tmp = np.array(tmp).sum() / (span / 252)
            gas_hisvol.iloc[i] = np.sqrt(tmp)
        gas_hisvol = gas_hisvol.dropna()
        self.gas_hisvol = gas_hisvol

        y = np.fft.fft(gas_hisvol)
        y_real = np.real(y).squeeze()
        y_imag = np.imag(y).squeeze()
        freq = np.fft.fftfreq(y.shape[0])
        self.y_real = y_real
        self.freq = freq

        self.num_freq = 10

        sep = 50
        self.freq_ = freq[y_real.argsort()[::sep][-self.num_freq:]]
        time = np.linspace(0, len(gas_hisvol) / 365, len(gas_hisvol))
        self.midtime = np.median(time)
        endtime = time.max()
        self.time_shift = time - endtime

    def __call__(self, x):
        fitted = 4
        for i in range(self.num_freq):
            fitted += 1 / self.num_freq * \
                      np.cos(2 * pi * (self.freq_[i]) * (x + self.midtime))
        return fitted

    def plotFreq(self):
        # frequency plot
        plt.figure()
        plt.plot(self.freq, self.y_real)
        plt.show()

    def plotAgainstReal(self):
        fig = plt.figure()
        ax = fig.subplots(1)
        x = np.linspace(-2, 1, 420)
        gas_hisvol_fitted = self(x)
        plt.plot(self.time_shift, np.array(self.gas_hisvol['Vol']))
        plt.plot(x, gas_hisvol_fitted)
        plt.show()


class MonteCarloSimulator:
    def __init__(self, model_params, cur_mkt_val, maturity):
        '''
        model_params: theta (array of func of t, days. [theta_e, theta_g]), rho, alpha(array of size 2,
        [alpha_e, alpha_g]), vol, win_len, log_rv
        cur_mkt_val: e0, g0
        maturity: T
        sim_info: num_sim, n_steps
        '''

        self._model_params = model_params
        self._cur_mkt_val = cur_mkt_val
        self._maturity = maturity

    def sim_path(self, num_sim=10000, asofdate=None, todate=None, use_fourier=False):
        def rolling_vol(i_day, mat, win_len, log_rv=False):
            if i_day + 1 < win_len:
                if log_rv:
                    return np.sqrt(
                        np.sum(np.diff(np.log(mat[:, 0:i_day]), axis=1) ** 2, axis=1) / (
                                    i_day + 1) * 250)  # Not sure here
                return np.sqrt(
                    np.sum(np.diff(mat[:, 0:i_day], axis=1) ** 2, axis=1) / (i_day + 1) * 250)  # Not sure here
            else:
                if log_rv:
                    return np.sqrt(
                        np.sum(np.diff(np.log(mat[:, i_day + 1 - win_len:i_day + 1]), axis=1) ** 2,
                               axis=1) / win_len * 250)
                return np.sqrt(
                    np.sum(np.diff(mat[:, i_day + 1 - win_len:i_day + 1], axis=1) ** 2, axis=1) / win_len * 250)

        e0, g0 = self._cur_mkt_val
        theta, vol, win_len, log_rv = self._model_params
        alpha_e, alpha_g = 5.2, 4.4
        rho = 0.9
        theta_e, theta_g = theta
        vol_e, vol_g = vol

        # dates
        if asofdate is None:
            asofdate = 0
        if todate is None:
            todate = self._maturity

        delta_t = 1 / 240
        n_steps = todate - asofdate
        et = np.ones((num_sim, n_steps + 1)) * e0
        gt = np.ones((num_sim, n_steps + 1)) * g0
        Z1 = norm.rvs(size=num_sim * n_steps).reshape(num_sim, n_steps)
        Z2 = norm.rvs(size=num_sim * n_steps).reshape(num_sim, n_steps)
        B1 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        curdate = asofdate
        for i in range(n_steps):
            curdate += 1
            et[:, i + 1] = et[:, i] + alpha_e * (theta_e(curdate) - et[:, i]) * delta_t + \
                           rolling_vol(i, et, win_len, log_rv) * \
                           ((et[:, i] - 1) * log_rv + 1) * np.sqrt(delta_t) * Z1[:, i]
            if use_fourier:
                gt[:, i + 1] = gt[:, i] + alpha_g * (theta_g(curdate) - gt[:, i]) * \
                               delta_t + vol_g(curdate / 240) * np.sqrt(delta_t) * B1[:, i]  # fourier use year as unit
            else:
                gt[:, i + 1] = gt[:, i] + alpha_g * (theta_g(curdate) - gt[:, i]) * delta_t + \
                               rolling_vol(i, gt, win_len, log_rv) * \
                               ((gt[:, i] - 1) * log_rv + 1) * np.sqrt(delta_t) * B1[:, i]
            et[:, i + 1] = np.abs(et[:, i + 1])
            gt[:, i + 1] = np.abs(gt[:, i + 1])
        return et, gt


def futures(history):
    '''
    Monthly Block Futures Contract：payoff is the expected value of monthly block spot price
    '''
    et, gt = history
    et_lst = np.zeros(13)
    gt_lst = np.zeros(13)
    for i in range(13):
        et_lst[i] = np.mean(np.mean(et[:, i*20:(i+1)*20+1], axis=1))
        gt_lst[i] = np.mean(np.mean(gt[:, i*20:(i+1)*20+1], axis=1))
    return et_lst, gt_lst


def euro_call(history, K, r):
    '''
    T: expiry
    r: yield_curve
    K: [Ke, Kg] can just input the spot, translation to forward is conducted within the function
    Monthly Block Call Options: underlying is the futures and option expiry = futures delivery.. well it's the fucking spot
    '''
    et, gt = history
    ke, kg = K
    et_lst = np.zeros(12)
    gt_lst = np.zeros(12)
    for i in range(12):
        tau = ((i+1)*20)/240
        et_lst[i] = np.mean(np.maximum(et[:,(i+1)*20] - ke*np.exp(r(tau)*tau), 0))*np.exp(-r(tau)*tau)
        gt_lst[i] = np.mean(np.maximum(gt[:,(i+1)*20] - kg*np.exp(r(tau)*tau), 0))*np.exp(-r(tau)*tau)
    return et_lst, gt_lst


def yield_curve(t):
    '''
    t: annualized maturity: 1mo = 1/12
    '''
    x = np.linspace(0, 1+1/12, num=14)
    y = np.array([0, 0.0300, 0.0320, 0.0320, 0.0315, 0.0320, 0.0320, 0.0320, 0.0330,
                  0.0330, 0.0330, 0.0330, 0.0330, 0.0330])
    return interp1d(x, y, kind='cubic')(np.array([t]))[0]

vol_fourier = VolGFourier()

class PathGenerator:
    def __init__(self):
        month_start = np.arange(0, 261)[::20]
        mu0 = np.array([95.76548345, 89.52283873, 95.35215501, 104.46874005, 94.6203973,
                        80.91702108, 100.07035029, 93.69413937, 100.4471256, 100.71685473,
                        99.78141199, 96.19445081, 79.62785258, 76.88922716, 9.45413411,
                        9.93164887, 10.68021599, 10.53612154, 10.75736881, 11.92659786,
                        10.16327577, 12.79479404, 9.80340575, 13.90669016, 9.79955108,
                        13.78518266, 6.879269, 6.98612125])
        theta_e_grid = mu0[:14]
        theta_g_grid = mu0[14:]
        theta_e = interp1d(month_start, theta_e_grid, kind='linear')
        theta_g = interp1d(month_start, theta_g_grid, kind='linear')

        vol_g_fourier = vol_fourier
        model_params = [np.array([theta_e, theta_g]), (None, vol_g_fourier), 20, True]
        cur_mkt_val = [82, 9.52]
        maturity = 260
        self.mc = MonteCarloSimulator(model_params, cur_mkt_val, maturity)

    def getPath(self, num_sim, asofdate=None, todate=None, spot=None):
        default_cur_mkt_val = self.mc._cur_mkt_val
        if spot is not None:
            if len(spot) != 2:
                raise ValueError("The spot argument should be a list of two values")

            self.mc._cur_mkt_val = spot

        history = self.mc.sim_path(num_sim=num_sim, asofdate=asofdate, todate=todate, use_fourier=True)
        self.mc._cur_mkt_val = default_cur_mkt_val
        return history


if __name__ == '__main__':
    vol_g_fourier = VolGFourier()
    # vol_g_fourier.plotAgainstReal()

    pg = PathGenerator()
    his = pg.getPath(10, spot=[80, 10])
    plt.plot(his[0].T)
    plt.show()
    print(0)














