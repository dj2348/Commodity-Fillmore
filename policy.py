import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from pricer import *


class DispatchSimulator:
	def __init__(self, threshold_on=4, threshold_off=0):
		'''
		threshold_on: threshold for operating (from off to on), measured by spark spread
		threshold_off: threshold for not operating (from on to off), measured by spark spread
		
		heatrate: number of mmBTU of gas needed to generate 1 MWH electricity
		capacity: capacity of power plant as long as it's on, 500 megawatts/h
		cost_start: additional start up charge on each MWH during the first week it restarted
		cost_var: variable cost per MWH when operating
		wait_off: extra days to stay off after it's turned off
		
		OAS_h: Option adjusted spread of current one-year tolling agreement 
		OAS_l: Lowest OAS discount factor (s.t. the price will be highest) Brad willing to pay
		
		r: yield for RN dynamics, same as the objective dynamics
		path: path generator
		'''
		self.threshold_on = threshold_on
		self.threshold_off = threshold_off
		
		# print("Operating threshold now: ", self.threshold_on)
		# print("Not perating threshold now: ", self.threshold_off)
		
		self.heatrate = 8
		self.capacity = 500
		self.cost_start = 5
		self.cost_var = 2
		self.waittime = 5
		
		
		self.OAS_h = 0.05
		self.OAS_l = 0.02
		
		self.r = yield_curve
		self.path = PathGenerator()

		self.cost_share_ratio = 0.5
		
	def sparkspread(self, n):
		'''
		return spark spread path
		'''
		power, gas = self.path.getPath(n)    
		return power - self.heatrate * gas, power, gas
		
	def dispatch(self, n, verbose=False):
		'''
		state: bool, 1 - Power station is on, 0 - Power station is off
		wait: Number of days that still need to be wait to turn on the power station
		restart: [bool, number of days left in restart priod]
				1 - Power station is in the five-day restart period, 0 - not
		'''
		path, power_path, gas_path = self.sparkspread(n)
		cashflow_list = []
		undiscount_list = []
		
		power_list = []
		power_undist_list = []

		gas_list = []
		gas_undist_list = []

		for n in range(len(path)):
			path_cashflow = 0
			path_undist_cashflow = 0

			power_cashflow = 0
			power_undist_cashflow = 0

			gas_cashflow = 0
			gas_undist_cashflow = 0

			state = 1
			wait = 0
			restart = [0, 0]

			for i in range(20, len(path[n])):
				spread = path[n][i]
				discount = self.r(i/240) + self.OAS_l

				undiscount_cf = state * (spread - restart[0] * 5 - 2) * self.capacity * 16
				variable_cost = state * (restart[0] * 5 + 2)
				power_comp = state * (power_path[n][i] - (1 - self.cost_share_ratio) * variable_cost) * self.capacity * 16
				gas_comp = state * (- self.heatrate * gas_path[n][i] - self.cost_share_ratio * variable_cost) * self.capacity * 16  

				path_cashflow += undiscount_cf * exp(-discount * i/240)
				path_undist_cashflow += undiscount_cf

				power_cashflow += power_comp * exp(-discount * i/240)
				power_undist_cashflow += power_comp

				gas_cashflow += gas_comp * exp(-discount * i/240)
				gas_undist_cashflow += gas_comp


				if verbose:
					print('Spread today: ', spread)
					print('On or off: ', state)
					print('Restart: ', restart)
					print('Wait days: ', wait)
					print('Profit today: ', state * (spread - restart[0] * 5 - 2) * self.capacity * 16)
					print('After discount: ', state * (spread - restart[0] * 5 - 2) * self.capacity * 16 * exp(-discount * i/240))
					print('Power Component: ', power_comp)
					print('Gas Component: ', gas_comp)
					print(' ')

				if wait > 0:
					wait -= 1

				if restart[0] == 1:
					restart[1] -= 1
					if restart[1] == 0:
						restart = [0, 0]

				if state == 1 and spread <= self.threshold_off:
					# Turn off the plant
					state = 0
					wait = 6
					restart = [0, 0]

				if state == 0 and wait == 0 and spread >= self.threshold_on:
					# Turn on the plant
					state = 1
					restart = [1, 5]

			cashflow_list.append(path_cashflow)
			undiscount_list.append(undiscount_cf)
			
			power_list.append(power_cashflow)
			power_undist_list.append(power_undist_cashflow)

			gas_list.append(gas_cashflow)
			gas_undist_list.append(gas_undist_cashflow)
			
		return (cashflow_list, undiscount_list, 
				power_list, power_undist_list, 
				gas_list, gas_undist_list)
	
	def value(self, n):
		'''
		return the mean discounted value and associated std
		of the plant cash flow under sub-optimal policy
		
		n: number of path used in valuation
		'''
		result, _, power, _, gas, _  = self.dispatch(n)		
		
		# print(" ")
		# print("Total:{}/{}, Power:{}/{}, Gas:{}/{}, Gas Min: {}".format(np.mean(result), np.std(result) / np.sqrt(n),
		# 		np.mean(power), np.std(power) / np.sqrt(n),
		# 		np.mean(gas), np.std(gas) / np.sqrt(n), np.min(gas)))

		return (np.mean(result), np.std(result) / np.sqrt(n),
				np.mean(power), np.std(power) / np.sqrt(n),
				np.mean(gas), np.std(gas) / np.sqrt(n))


# if __name__ == '__main__':
# 	sim = DispatchSimulator()
# 	sim.value(500)
