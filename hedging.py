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
        self.spread_short_threshold = 1
        self.spread_hedge_ratio = 20


    def near_futures_price(self, t, spot):
        '''
        Calculate the price of nearest futures contract
        :param t: int, the number of days since start of simulation
        :param spot: n by 2 array, the current power and gas prices
        :return futures_price: n by 2 array, the current power and gas futures prices
        '''

        if t == 260:
            return spot.T

        n = spot.shape[1]
        futures_price = np.zeros((n,2))
        for i in range(n):
            power_path, gas_path = self.path.getPath(10*n, t, (t//20+1)*20, spot[:,i])
            futures_price[i,] = [np.mean(power_path[:,-1]), np.mean(gas_path[:,-1])]
        return futures_price

    def hedging(self, n):
        path, power_path, gas_path = self.sparkspread(n)
        Nt = path.shape[1]

        # state variable for each path
        state = np.ones(n)*self.multiplier
        count = np.zeros(n)
        wait = np.zeros(n)
        restart = np.zeros((n,2))
        path_cashflow = np.zeros((n,Nt-21))
        path_hedging_cashflow = np.zeros((n,Nt-21))
        hedging_res = np.zeros((n,Nt-21))

        # First column indicates whether short gas; the second indicates whether short spread
        # value of 1 means currently short, value of 0 means no exposure
        hedging_status = np.zeros((n,2)).astype('bool')
        # indicates the position of spread futures, -1 short, 0 no position, 1 long
        hedging_spread_position = np.zeros(n)
        last_gas = np.zeros(n)
        last_spread = np.zeros(n)
        gas_futures_path = np.zeros((n,Nt-21))
        power_futures_path = np.zeros((n,Nt-21))

        for i in range(21, Nt):
            print(f'Day: {i-21}')
            count += (state==self.multiplier)
            op_flag = count<=self.count_days

            cur_spread = path[:,i]
            discount = self.r(i/240) + self.OAS_l

            # undiscount_cf = state*(cur_spread - restart[:,0]*5 - 2)*self.capacity*16
            variable_cost = state*(restart[:,0]*5 + 2)
            # power_comp = state*(power_path[:,i] - (1 - self.cost_share_ratio)*variable_cost)*self.capacity*16
            # gas_comp = state*(- self.heatrate*gas_path[:,i] - self.cost_share_ratio*variable_cost)*self.capacity*16

            undiscount_cf = state*(self.heatrate*gas_path[:,i] + self.cost_share_ratio*variable_cost)*\
                            self.capacity*16

            ### Hedging
            futures_price = self.near_futures_price(i, np.vstack((power_path[:,i],gas_path[:,i])))
            power_futures = futures_price[:,0]
            gas_futures = futures_price[:,1]
            spread_futures = power_futures - gas_futures*self.heatrate

            undiscount_hedging_cf = undiscount_cf.copy()
            # calculate gas futures pnl
            undiscount_hedging_cf[hedging_status[:, 0] & op_flag] -= self.multiplier*self.capacity*16*self.heatrate * \
                (gas_futures[hedging_status[:, 0] & op_flag] - last_gas[hedging_status[:, 0] & op_flag])
            hedging_res[:,i-21] = (undiscount_hedging_cf - undiscount_cf)*exp(-discount*i/240)
            # calculate spread futures pnl
            undiscount_hedging_cf[hedging_spread_position>0 & op_flag] += \
                self.multiplier*self.capacity*16*self.heatrate * self.spread_hedge_ratio * \
                (spread_futures[hedging_spread_position>0 & op_flag] - last_spread[hedging_spread_position>0 & op_flag])
            undiscount_hedging_cf[hedging_spread_position<0 & op_flag] -= \
                self.multiplier*self.capacity*16*self.heatrate*self.spread_hedge_ratio* \
                (spread_futures[hedging_spread_position<0 & op_flag] - last_spread[hedging_spread_position<0 & op_flag])
            # update hedging status
            target_hedging_status = np.array([state==self.multiplier, state==0]).T
            hedging_status = target_hedging_status
            hedging_spread_position[~target_hedging_status[:,1]] = 0
            hedging_spread_position[target_hedging_status[:, 1]] = 1
            # change spread long position to short position if spread back to short threshold
            hedging_spread_position[target_hedging_status[:,1] & (cur_spread>self.spread_short_threshold)] = -1
            # ## exit spread position if reaching cut loss level
            # hedging_spread_position

            # close out if reaching the last day of contract or preset cap
            close_flag = np.ones(n).astype('int')*(i==(Nt-1))|(count==self.count_days)
            hedging_status[close_flag, :] = [0,0]

            last_gas = gas_futures
            last_spread = spread_futures

            gas_futures_path[:,i-21] = gas_futures
            power_futures_path[:, i - 21] = power_futures

            path_cashflow[op_flag,i-21] = undiscount_cf[op_flag]*exp(-discount*i/240)
            path_hedging_cashflow[op_flag,i-21] = undiscount_hedging_cf[op_flag]*exp(-discount*i/240)


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

        spread_futures_path = power_futures_path - gas_futures_path*self.heatrate

        return path_cashflow, path_hedging_cashflow, hedging_res


if __name__== '__main__':
    start = time.time()
    hedge = HedgingSimulator()
    cash, hedged_cash, hedging_gas_futures = hedge.hedging(10)
    end = time.time()
    hedging_spread_futures = hedged_cash - cash - hedging_gas_futures
    print(f'Hedging takes {end-start} s')
    print(f'correlation of gas futures: {np.corrcoef(np.sum(cash,axis=1),np.sum(hedging_gas_futures,axis=1))}')
    print(f'correlation of spread futures: {np.corrcoef(np.sum(cash, axis=1), np.sum(hedging_spread_futures, axis=1))}')
    print(f'Unhedged Cashflow: {np.mean(np.sum(cash,axis=1))}({np.std(np.sum(cash,axis=1))/10**0.5})')
    print(f'Hedged Cashflow: {np.mean(np.sum(hedged_cash,axis=1))}({np.std(np.sum(hedged_cash,axis=1))/10**0.5})')

    total_cash = np.sum(cash,axis=1)
    total_hedged_cash = np.sum(hedged_cash,axis=1)
    plt.hist([total_cash, total_hedged_cash], bins=20)
    plt.legend(['cash', 'hedged'])
    plt.show()
    print(0)