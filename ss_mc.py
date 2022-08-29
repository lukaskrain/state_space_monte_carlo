###########################################################
#
# Class that performs State Space Monte Carlo Simulations
#
###########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy
from scipy import stats
from hmmlearn import hmm
from sstudentt import SST
from fitter import Fitter
import math

class SS_MC:
    
    def __init__(self, dist=["norm", "lognorm", "t", "laplace", "cauchy", "skewcauchy"]):
        self.possible_dist = dist
        self.state_dist = {}
    
    def fit_states(self, ret, n, iterations=500, rescale=False, crit="bic"):
        self.n_states = n
        self.ret = ret.iloc[:, 0].values
        self.dates = ret.index
        if rescale:
            ret /= ret.std()
        hmm_model = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=iterations).fit(ret)
        self.states = hmm_model.predict(ret)
        self.state_prob = hmm_model.predict_proba(ret)
        self.transition_matrix = pd.crosstab(pd.Series(self.states[1:],name='next state'),
                                             pd.Series(self.states[:-1],name='current state'),
                                             normalize=1)
        for i in range(n):
            self.estimate_dist(state_num = i, crit = crit)
    
    def estimate_dist(self, state_num, crit):
        f = Fitter(self.ret[self.states == state_num], distributions = self.possible_dist)
        f.fit()
        self.state_dist[state_num] = f.get_best(crit)
    
    def plot_states(self):
        try:
            self.n_states
        except:
            print("Must fit states first.")
            return None
        colormap = np.array(['red', 'green', 'blue', 'yellow', 'grey', 'black', 'brown'])
        if self.n_states > len(colormap):
            print("Not enough colors. Some colors may be used for different states.")
        plt.scatter(self.dates, self.ret, c=colormap[self.states])
        plt.show()
    
    def estimate_scauchy_states(self):
        try:
            self.n_states
        except:
            print("Must fit states first")
            return None
        self.scauchy_param = {}
        for i in range(self.n_states):
            state_ret = self.ret[self.states == i]
            f = Fitter(state_ret, distributions = ["skewcauchy"])
            f.fit()
            est_params = f.get_best()["skewcauchy"]
            self.scauchy_param[i] = est_params
    
    def mc_sim(self, n, T, dist="normal", **params):
        if dist == "normal":
            try:
                rets = pd.DataFrame((np.random.normal(params["mu"],
                                                      params["sd"],
                                                      size = n*T)+1).reshape((T, n))).cumprod()-1
            except:
                print("Missing keys for selected distribution.")
                return None
        if dist == "t":
            try:
                rets = pd.DataFrame((np.random.standard_t(params["df"],
                                                          size = n*T)+1).reshape((T, n))).cumprod()-1
            except:
                print("Missing keys for selected distribution.")
                return None
        return rets
    
    def state_rv(self, state):
        distribution = [*self.state_dist[state]][0]
        parameters = [p for p in self.state_dist[state].get(distribution).values()]
        rv = eval("scipy.stats." + distribution)
        return rv.rvs(*parameters, size=1)
    
    def fitted_ss_mc(self, n, T, initial_state=0):
        try:
            self.states
        except:
            print("Must fit states first")
            return None
        simulated_rets = np.zeros(n*T).reshape(T, n)
        states = [initial_state for _ in range(n)]
        for t in range(T):
            new_rets = np.concatenate([self.state_rv(state) for state in states])
            simulated_rets[t] = new_rets
            states = [np.random.choice(list(range(len(self.transition_matrix))),
                                       p=self.transition_matrix[state]) for state in states]
        self.sim_rets = pd.DataFrame(simulated_rets+1).cumprod()-1
            
    def monte_carlo(self, n, T, densities, transition_matrix, initial_state=0):
        '''
        {0: {"norm": {"loc": 0, "scale": 0.03}},
        1: {"laplace": {"loc": 0.04, "scale": 0.1}}}
        '''
        if len(densities) != len(transition_matrix):
            print("Mismatch between number of densities and states.",
                  "\n",
                  "Tried to assign %i densities to %i states"%(len(densities), states_num))
            return None
        simulated_rets = np.zeros(n*T).reshape(T, n)
        states = [initial_state for _ in range(n)]
        dist = []
        for i in range(len(densities)):
            distribution = [*densities[i]][0]
            parameters = [p for p in densities[i].get(distribution).values()]
            dist.append([eval("scipy.stats."+distribution), parameters])
        for t in range(T):
            new_rets = np.concatenate([dist[state][0].rvs(*dist[state][1], size=1) for state in states])
            simulated_rets[t] = new_rets
            states = [np.random.choice([0,1], p=transition_matrix[state]) for state in states]
        self.sim_rets = pd.DataFrame(simulated_rets+1).cumprod()-1
        
    def plot_montecarlo(self):
        try:
            plt.plot(self.sim_rets)
            plt.title("Monte Carlo Simulation of Return Paths")
            plt.xlabel("Time Periods")
            plt.ylabel("Return in Per Cent")
            plt.show()
        except:
            print("Need to simulate returns first.")
            return None
    
    def maturity_cdf(self, crit="aic"):
        try:
            f = Fitter(self.sim_rets.iloc[-1, :], distributions = self.possible_dist)
        except:
            print("Need to simulate returns first.")
            return None
        f.fit()
        dist = [*f.get_best(crit)][0]
        return eval("scipy.stats." + dist), f.get_best(crit)[dist]