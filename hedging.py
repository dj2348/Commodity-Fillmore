import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from pricer import *
from policy import *

class HedgingSimulator(DispatchSimulator):
    '''
    Simulator for hedging analysis; variables are in general named as those in DispatchSimulator
    '''
    def __init__(self):
        super().__init__()
        self.spread_short_threshold = 4


    def near_futures_price(self, t, spot):
        '''
        Calculate the price of nearest futures contract
        :param t: int, the number of days since start of simulation
        :param spot: n by 2 array, the current power and gas prices
        :return futures_price: n by 2 array, the current power and gas futures prices
        '''

        n = spot.shape[0]
        futures_price = np.zeros((n,2))
        for i in range(n):
            _, power_path, gas_path = self.path.getPath(n, t, (t//20+1)*20, spot[i,:])
            futures_price[i,] = [np.mean(power_path, np.mean(gas_path))]
        return futures_price

    def hedging(self, n):
        path, power_path, gas_path = self.sparkspread(n)
        Nt = path.shape[1]

        # state variable for each path
        state = np.ones(n)*self.multiplier
        count = np.zeros(n)
        restart = np.zeros((n,2))
        path_cashflow = np.zeros(n)
        path_hedging_cashflow = np.zeros(n)
        wait = np.zeros(n)

        # First column indicates whether short gas; the second indicates whether short spread
        # value of 1 means currently short, value of 0 means no exposure
        hedging_status = np.zeros((n,2))
        last_gas = np.zeros(n)
        last_spread = np.zeros(n)

        for i in range(21, Nt):
            count += (state==self.multiplier)
            op_flag = count<=self.count_days

            cur_spread = path[:,i]
            discount = self.r(i/240) + self.OAS_l

            undiscount_cf = state*(cur_spread - restart[:,0]*5 - 2)*self.capacity*16
            variable_cost = state*(restart[:,0]*5 + 2)
            power_comp = state*(power_path[:,i] - (1 - self.cost_share_ratio)*variable_cost)*self.capacity*16
            gas_comp = state*(- self.heatrate*gas_path[:,i] - self.cost_share_ratio*variable_cost)*self.capacity*16


            ### Hedging
            futures_price = self.near_futures_price(i, np.hstack((power_path[:,i],gas_path[:,i])))
            gas_futures = futures_price[:,0]
            power_futures = futures_price[:,1]
            spread_futures = power_futures - gas_futures

            target_hedging_status = np.array([state==self.multiplier, cur_spread<self.spread_short_threshold])

            undiscount_hedging_cf = undiscount_cf
            # trade gas futures
            undiscount_hedging_cf[target_hedging_status[:,0] > hedging_status[:,0] & op_flag] += gas_futures[
                target_hedging_status[:,0] > hedging_status[:,0] & op_flag]
            undiscount_hedging_cf[target_hedging_status[:,0] < hedging_status[:,0] & op_flag] -= gas_futures[
                target_hedging_status[:,0] < hedging_status[:,0] & op_flag]
            undiscount_hedging_cf[target_hedging_status[:, 0] & hedging_status[:, 0] & op_flag] -= \
                (gas_futures[target_hedging_status[:, 0] & hedging_status[:, 0] & op_flag] -
                last_gas[target_hedging_status[:, 0] & hedging_status[:, 0] & op_flag])
            # trade spread
            undiscount_hedging_cf[target_hedging_status[:, 1] > hedging_status[:, 1] & op_flag] += spread_futures[
                target_hedging_status[:, 1] > hedging_status[:, 1] & op_flag]
            undiscount_hedging_cf[target_hedging_status[:, 1] < hedging_status[:, 1] & op_flag] -= spread_futures[
                target_hedging_status[:, 1] < hedging_status[:, 1] & op_flag]
            undiscount_hedging_cf[target_hedging_status[:, 1] & hedging_status[:, 1] & op_flag] -= \
                (spread_futures[target_hedging_status[:, 1] & hedging_status[:, 1] & op_flag] -
                 last_spread[target_hedging_status[:, 1] & hedging_status[:, 1] & op_flag])
            # close out if reaching the last day of contract or preset cap
            close_flag = np.ones(n).astype('int')*(i==(Nt-1))|(count==self.count_days)
            undiscount_hedging_cf[target_hedging_status[:, 0] & op_flag & close_flag] += gas_futures[
                target_hedging_status[:, 0] & op_flag & close_flag]
            undiscount_hedging_cf[target_hedging_status[:, 1] & op_flag & close_flag] += spread_futures[
                target_hedging_status[:, 1] & op_flag & close_flag]

            last_gas = gas_futures
            last_spread = spread_futures

            # TODO: specify the exact cashflow to export
            path_cashflow[op_flag] += undiscount_cf[op_flag]*exp(-discount[op_flag]*i/240)
            path_hedging_cashflow[op_flag] += undiscount_hedging_cf[op_flag]*exp(-discount[op_flag]*i/240)


            ### Virtual Dispatch Decision & Operation
            wait = np.maximum(wait-1, 0)

            restart[restart[:,0]==1,1] -= 1
            restart[restart[:,1]==0,:] = [0,0]

            # turn off the plant
            turn_off_flag = (state==self.multiplier) & (cur_spread <= self.threshold_off)
            state[turn_off_flag] = 0
            wait[turn_off_flag] = 6
            restart[turn_off_flag,:] = [0,0]

            # turn on the plant
            turn_on_flag = (state==0) & (cur_spread >= self.threshold_on) & (wait==0)
            state[turn_on_flag] = self.multiplier
            restart[turn_on_flag, :] = [1, 5]

        return path_cashflow, path_hedging_cashflow