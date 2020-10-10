import numpy as np
from numpy import exp, sqrt, log, pi
from numpy.fft import fft
import pandas as pd
import matplotlib.pyplot as plt

lam_gas = 4.4
span = 126
delta_t = 1/252

gas_hisspot = pd.read_csv('Henry_Hub_Natural_Gas_Spot_Price.csv')
gas_hisspot.columns = ['Day', 'Spot']
gas_hisspot['Day'] = pd.to_datetime(gas_hisspot['Day'])
gas_hisspot = gas_hisspot.set_index('Day').sort_index()
gas_hisspot = gas_hisspot.loc['2006-01-01':'2008-04-01']

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

y = np.fft.fft(gas_hisvol.loc['2006-01-01':'2008-04-01'])
y_real = np.real(y).squeeze()
y_imag = np.imag(y).squeeze()
freq = np.fft.fftfreq(y.shape[0])

# frequency plot
''' 
plt.figure()
plt.plot(freq, y_real)
plt.show()
'''

n = 3
freq_ = freq[y_real.argsort()[::20][-n:]]


x = np.linspace(0, len(y)/252, len(y))


def volatility(x):
    fitted = 5
    for i in range(n):
        fitted += 0.5 * np.cos(2*pi*(freq_[i]+0.13) * (x + 1.73015873) - 1.6)
    return fitted

x = np.linspace(0, len(gas_hisvol['Vol'])/252, len(gas_hisvol['Vol'])) - 1.73015873

gas_hisvol_fitted = volatility(x)

fig = plt.figure()
ax = fig.subplots(1)
plt.plot(x, np.array(gas_hisvol['Vol']))
plt.plot(x, gas_hisvol_fitted)
plt.show()



print(0)

