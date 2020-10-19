import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
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
        self.cost_share_ratio = 0.5
        self.spread_short_threshold = 1
        self.spread_hedge_ratio = 20
        self.count_days = 240

    def near_futures_price(self, t, spot):
        '''
        Calculate the price of nearest futures contract
        :param t: int, the number of days since start of simulation
        :param spot: n by 2 array, the current power and gas prices
        :return futures_price: n by 2 array, the current power and gas futures prices
        '''

        if t > 240:
            raise ValueError(f"No Futures Contract Available at Time {t}")

        n = spot.shape[1]
        futures_price = np.zeros((n, 2))
        for i in range(n):
            power_path, gas_path = self.path.getPath(1000, ((t-1)//20 + 1)*20+1, ((t-1)//20 + 2)*20, spot[:, i])
            futures_price[i,] = [np.mean(power_path), np.mean(gas_path)]
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
        hedging_res_gas = np.zeros((n,Nt-21))
        variable_cost_cum = np.zeros(n)

        # indicates the position of spread futures, -1 short, 0 no position, 1 long
        hedging_spread_position = np.zeros(n)
        last_gas = np.zeros(n)
        last_spread = np.zeros(n)
        gas_futures_path = np.zeros((n,Nt-41))
        power_futures_path = np.zeros((n,Nt-41))

        # calculate gas futrues price for the first month
        futures_price = self.near_futures_price(20, np.vstack((power_path[:, 20], gas_path[:, 20])))
        hedged_gas_price = futures_price[:,1]

        # indicates whether purchasing gas futures for this month
        # only trade futures when spread before futures month is positive
        lockin_status = path[:,20]>0

        for i in range(21, Nt):
            print(f'Day: {i-21}')
            count += (state==self.multiplier)
            op_flag = count<=self.count_days

            cur_spread = path[:,i]
            discount = self.r(i/240) - self.OAS_l

            # undiscount_cf = state*(cur_spread - restart[:,0]*5 - 2)*self.capacity*16
            variable_cost = restart[:,0]*5 + 2
            # power_comp = state*(power_path[:,i] - (1 - self.cost_share_ratio)*variable_cost)*self.capacity*16
            # gas_comp = state*(- self.heatrate*gas_path[:,i] - self.cost_share_ratio*variable_cost)*self.capacity*16

            undiscount_cf = state*(self.heatrate*gas_path[:,i] + self.cost_share_ratio*variable_cost)*\
                            self.capacity*16
            # variable_cost_cum += state*variable_cost*self.capacity*16

            ### Hedging
            undiscount_hedging_cf = undiscount_cf.copy()

            if i<=240:
                # calculate spread futures pnl
                futures_price = self.near_futures_price(i, np.vstack((power_path[:, i], gas_path[:, i])))
                power_futures = futures_price[:, 0]
                gas_futures = futures_price[:, 1]
                spread_futures = power_futures - gas_futures*self.heatrate
                undiscount_hedging_cf[hedging_spread_position>0 & op_flag] += \
                    self.multiplier*self.capacity*16*self.heatrate * self.spread_hedge_ratio * \
                    (spread_futures[hedging_spread_position>0 & op_flag] - last_spread[hedging_spread_position>0 & op_flag])
                undiscount_hedging_cf[hedging_spread_position<0 & op_flag] -= \
                    self.multiplier*self.capacity*16*self.heatrate*self.spread_hedge_ratio* \
                    (spread_futures[hedging_spread_position<0 & op_flag] - last_spread[hedging_spread_position<0 & op_flag])
                # update hedging status
                # target_hedging_status = np.array([state==self.multiplier, state==0]).T
                target_hedging_status = state==0
                hedging_status = target_hedging_status
                hedging_spread_position[~target_hedging_status] = 0
                hedging_spread_position[target_hedging_status] = 1
                # change spread long position to short position if spread back to short threshold
                hedging_spread_position[target_hedging_status & (cur_spread>self.spread_short_threshold)] = -1
                # ## exit spread position if reaching cut loss level
                # hedging_spread_position

                # close out if reaching the last day of contract or preset cap
                close_flag = np.ones(n).astype('int')*(i%20==0)|(count>=self.count_days)
                hedging_status[close_flag] = 0

                last_spread = spread_futures
                # update locked-in gas prices for next month
                if i%20==0:
                    hedged_gas_price = gas_futures
                    lockin_status = cur_spread>0

                gas_futures_path[:,i-21] = gas_futures
                power_futures_path[:, i - 21] = power_futures

                hedging_res[:,i-21] = (undiscount_hedging_cf - undiscount_cf)*exp(-discount*i/240)

            # calculate gas futures pnl
            undiscount_hedging_cf[lockin_status&op_flag] -= self.multiplier*self.capacity*16*self.heatrate*\
                                                    (hedged_gas_price[lockin_status&op_flag] - gas_path[lockin_status&op_flag, i])
            hedging_res_gas[lockin_status&op_flag,i-21] = -exp(-discount*i/240)*self.multiplier*self.capacity*16*self.heatrate*\
                                                    (hedged_gas_price[lockin_status&op_flag] - gas_path[lockin_status&op_flag, i])


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

        return path_cashflow, path_hedging_cashflow, hedging_res, hedging_res_gas


if __name__== '__main__':
    n = 1000
    start = time.time()
    hedge = HedgingSimulator()
    cash, hedged_cash, hedging_spread_futures, hedging_gas_futures = hedge.hedging(n)
    end = time.time()

    # # get data from stored file
    # file_name = "dynamic_hedge_res_1000_cost.xlsx"
    # cash = pd.read_excel(file_name, sheet_name="Unhedged Cash", index_col=0).values
    # hedged_cash = pd.read_excel(file_name, sheet_name="Hedged Cash", index_col=0).values
    # hedging_spread_futures = pd.read_excel(file_name, sheet_name="Spread Futures Component", index_col=0).values
    # hedging_gas_futures = pd.read_excel(file_name, sheet_name="Gas Futures Component", index_col=0).values

    # adjust discount rate
    # ori_cash = cash.copy()
    # ori_hedged_cash = hedged_cash.copy()
    adjust_discount = np.array([exp((2*hedge.OAS_l)*i/240) for i in range(21, 261)])
    cash *= adjust_discount
    hedged_cash *= adjust_discount
    hedging_spread_futures *= adjust_discount
    hedging_gas_futures *= adjust_discount
    # cash_ratio = np.sum(cash,axis=1)/np.sum(ori_cash,axis=1)
    # hedged_cash_ratio = np.sum(hedged_cash,axis=1)/np.sum(ori_hedged_cash,axis=1)
    # print(f'Cash:{np.mean(np.sum(cash,axis=1))}; Adjusted Cash:{np.mean(np.sum(ori_cash,axis=1))}')
    # print(f'Cash Ratio:{np.mean(cash_ratio)}({np.std(cash_ratio)}); Hedged Cash Ratio:{np.mean(hedged_cash_ratio)}({np.std(hedged_cash_ratio)})')


    # # Plan A
    # fixed_payment = 350000
    # discounted_fixed = np.ones(240)*fixed_payment
    # discount_vec = np.array([exp(-(hedge.r(i/240) - hedge.OAS_l)*i/240) for i in range(21, 261)])
    # cash -= discount_vec*fixed_payment
    # hedged_cash -= discount_vec *fixed_payment

    # # Plan B
    # count = pd.DataFrame(cash).ne(0).iloc[:,::-1].idxmax(axis=1).values + 1
    # fixed_payment = 320000
    # discount_vec = np.array([exp(-(hedge.r(i/240)-hedge.OAS_l)*i/240) for i in range(21,261)])
    # for i in range(n):
    #     cash[i,:count[i]] -= discount_vec[:count[i]]*fixed_payment
    #     hedged_cash[i,:count[i]] -= discount_vec[:count[i]]*fixed_payment

    # print(f'Hedging takes {end-start} s')
    print(f'correlation of spread futures: {np.corrcoef(np.sum(cash,axis=1), np.sum(hedging_spread_futures, axis=1))}')
    print(f'correlation of gas futures: {np.corrcoef(np.sum(cash, axis=1), np.sum(hedging_gas_futures, axis=1))}')
    print(f'Unhedged Cashflow: {np.mean(np.sum(cash,axis=1))}({np.std(np.sum(cash,axis=1))/n**0.5})')
    print(f'Hedged Cashflow: {np.mean(np.sum(hedged_cash,axis=1))}({np.std(np.sum(hedged_cash,axis=1))/n**0.5})')
    print(f'Probability of Positive PnL after Hedging: {np.mean(np.sum(hedged_cash,axis=1)>0)}; before Hedging:{np.mean(np.sum(cash,axis=1)>0)}')

    total_cash = np.sum(cash,axis=1)
    total_hedged_cash = np.sum(hedged_cash,axis=1)
    # kde_cash = stats.gaussian_kde(total_cash)
    # kde_hedged_cash = stats.gaussian_kde(total_hedged_cash)
    # total_cash_x = np.linspace(min(total_cash),max(total_cash),1000)
    # total_hedged_cash_x = np.linspace(min(total_hedged_cash),max(total_hedged_cash),1000)
    # plt.plot(total_cash_x,kde_cash(total_cash_x),'r-')
    # plt.plot(total_hedged_cash_x,kde_hedged_cash(total_hedged_cash_x),'b-')
    # plt.hist([total_cash, total_hedged_cash], bins=20, density=True)
    # plt.legend(['cash density','hedged cash density','cash', 'hedged'])
    # plt.show()

    VaR_cash = np.percentile(total_cash, 5)
    VaR_hedged_cash = np.percentile(total_hedged_cash, 5)
    print(f'5% VaR Unhedged:({VaR_cash}, Hedged:({VaR_hedged_cash})')

    plt.figure(figsize=(12, 8))
    plt.title("Structure A Total PnL Distribution", size=16)
    sns.distplot(total_cash)
    sns.distplot(total_hedged_cash)
    plt.xlabel("Discounted Total PnL")
    plt.axvline(VaR_cash, color='red', label='Unhedged 5% VaR')
    plt.axvline(VaR_hedged_cash, color='blue', label='Hedged 5% VaR')
    plt.legend(['Unhedged', 'Hedged', 'Unhedged 5% VaR', 'Hedged 5% VaR'])
    plt.show()


    # writer = pd.ExcelWriter('dynamic_hedge_res_1000_cost_120.xlsx', engine='xlsxwriter')
    # pd.DataFrame(cash).to_excel(writer, sheet_name='Unhedged Cash')
    # pd.DataFrame(hedged_cash).to_excel(writer, sheet_name='Hedged Cash')
    # pd.DataFrame(hedging_spread_futures).to_excel(writer, sheet_name='Spread Futures Component')
    # pd.DataFrame(hedging_gas_futures).to_excel(writer, sheet_name='Gas Futures Component')
    # writer.save()