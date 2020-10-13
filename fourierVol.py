import numpy as np
from numpy import exp, sqrt, log, pi
from numpy.fft import fft
import pandas as pd
import matplotlib.pyplot as plt

class VolGFourier:
    def __init__(self):
        lam_gas = 4.4
        span = 120
        delta_t = 1/252
        
        gas_hisspot = pd.read_excel('gas spot.xlsx')
        gas_hisspot.columns = ['Day', 'Spot']
        gas_hisspot['Day'] = pd.to_datetime(gas_hisspot['Day'])
        gas_hisspot = gas_hisspot.set_index('Day').sort_index()
        
        gas_hisspot_exp = gas_hisspot.copy()
        gas_hisspot_exp['t'] = np.arange(gas_hisspot.shape[0]) / 252
        gas_hisspot_exp['Spot'] = gas_hisspot_exp['Spot'] * exp(lam_gas*gas_hisspot_exp['t'])
        gas_hisspot_exp['diff'] = gas_hisspot_exp['Spot'] - gas_hisspot_exp['Spot'].shift(1)
        
        gas_hisret = gas_hisspot - gas_hisspot.shift(1)
        gas_hisret.columns = ['ret']
        gas_hisret['integral'] = (gas_hisspot + gas_hisspot.shift(1)) * delta_t / 2
        
        gas_hisvol = pd.DataFrame(index=gas_hisret.index, columns=['Vol'])
        for i in range(span, gas_hisret.shape[0]):
            tmp = (gas_hisret.iloc[i-span:i, 0] + lam_gas * gas_hisret.iloc[i-span:i, 1]) ** 2
            tmp = np.array(tmp).sum() / (span/252)
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
        time = np.linspace(0, len(gas_hisvol)/365, len(gas_hisvol))
        self.midtime = np.median(time)
        endtime = time.max()
        self.time_shift = time - endtime
    
    def __call__(self, x):
            fitted = 4
            for i in range(self.num_freq):
                fitted += 1 / self.num_freq * \
                          np.cos(2*pi * (self.freq_[i]) * (x + self.midtime))
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


if __name__ == '__main__':
    x = np.arange(-500, 240)
    vol_g_fourier = VolGFourier()
    vol_g_fourier.plotAgainstReal()



    print(0)

