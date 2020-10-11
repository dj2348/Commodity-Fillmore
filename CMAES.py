import numpy as np
from numpy import exp, sqrt, log, pi
import pandas as pd
import matplotlib.pyplot as plt

mu0 = np.random.randn(13)
sig0 = np.eye(13)

n_iter = 10000
lam = 10           # num of new points
k = 3              # num of elites
# start algo
mu, sig = mu0, sig0
for i in range(n_iter):
    new_set = np.random.multivariate_normal(mu, sig, lam)
    losses = []
    for params in new_set:
        losses += [np.random.randn()]#[aloss_function(params)]
    losses = np.array(losses)
    rank = np.argsort(losses)
    print('Elite losses', losses[rank[:k]].mean())
    elites = new_set[rank[:k]]
    elites = pd.DataFrame(elites)
    mu = elites.mean().to_numpy()
    sig = elites.cov().to_numpy()