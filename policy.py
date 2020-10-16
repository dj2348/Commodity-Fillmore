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
		
# 		print("Operating threshold now: ", self.threshold_on)
# 		print("Not perating threshold now: ", self.threshold_off)
		
		self.heatrate = 8
		self.capacity = 500
		self.cost_start = 5
		self.cost_var = 2
		self.waittime = 5
		
		
		self.OAS_h = 0.05
		self.OAS_l = 0.02
		
		self.r = yield_curve
		self.path = PathGenerator()
		
	def sparkspread(self, n):
		'''
		return spark spread path
		'''
		power, gas = self.path.getPath(n)    
		return power - self.heatrate * gas
		
	def dispatch(self, n, verbose=False):
		'''
		state: bool, 1 - Power station is on, 0 - Power station is off
		wait: Number of days that still need to be wait to turn on the power station
		restart: [bool, number of days left in restart priod]
				1 - Power station is in the five-day restart period, 0 - not
		
		return the cashflow and undiscounted cashflow for each path
		'''
		path = self.sparkspread(n)
		cashflow_list = []
		undiscount_list = []
		for n in range(len(path)):
			path_cashflow = 0
			state = 1
			wait = 0
			restart = [0, 0]
			for i in range(20, len(path[n])):
				spread = path[n][i]
				discount = self.r(i/240) + self.OAS_l
				undiscount_cf = state * (spread - restart[0] * 5 - 2) * self.capacity * 16
				path_cashflow += undiscount_cf * exp(-discount * i/240)
				
				if verbose:
					print('Spread today: ', spread)
					print('On or off: ', state)
					print('Restart: ', restart)
					print('Wait days: ', wait)
					print('Profit today: ', state * (spread - restart[0] * 5 - 2) * self.capacity * 16)
					print('After discount: ', state * (spread - restart[0] * 5 - 2) * self.capacity * 16 * exp(-discount * i/240))
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
			
		return cashflow_list, undiscount_list
	
	def value(self, n):
		'''
		return the mean discounted value and associated std
		of the plant cash flow under sub-optimal policy
		
		n: number of path used in valuation
		'''
		result, _ = self.dispatch(n)
		return np.mean(result), np.std(result) / np.sqrt(n)