import numpy as np
from numpy import exp, sqrt, log, pi
from numpy.fft import fft
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

lam_gas = 4.4
span = 120
delta_t = 1/252

gas_hisspot = pd.read_excel('gas spot.xlsx')
gas_hisspot.columns = ['Day', 'Spot']
gas_hisspot['Day'] = pd.to_datetime(gas_hisspot['Day'])
gas_hisspot = gas_hisspot.set_index('Day').sort_index()
#gas_hisspot = gas_hisspot.loc['2006-01-01':'2008-04-01']

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

print(gas_hisvol)
#gas_hisvol.plot()

y = np.fft.fft(gas_hisvol)
y_real = np.real(y).squeeze()
y_imag = np.imag(y).squeeze()
freq = np.fft.fftfreq(y.shape[0])

# frequency plot
''' 
plt.figure()
plt.plot(freq, y_real)
plt.show()
'''

n = 10
sep = 50
freq_ = freq[y_real.argsort()[::sep][-n:]]
time = np.linspace(0, len(gas_hisvol)/365, len(gas_hisvol))
midtime = np.median(time)
endtime = time.max()
time_shift = time - endtime
def volatility(x):
    fitted = 4
    for i in range(n):
        fitted += 1 / n * np.cos(2*pi * (freq_[i]) * (x + midtime))
    return fitted


x = np.linspace(-endtime, 1, 1000)
gas_hisvol_fitted = volatility(x)


fig = plt.figure()
ax = fig.subplots(1)
plt.plot(time_shift, np.array(gas_hisvol['Vol']))
plt.plot(x, gas_hisvol_fitted)
plt.show()



print(0)

