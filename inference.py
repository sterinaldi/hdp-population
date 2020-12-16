from scipy.special import logsumexp
from scipy.stats import norm
import cpnest.model
import sys
import os
import matplotlib.pyplot as plt
import corner
import itertools as it
import likelihood as lk
import math

class MassModel(cpnest.model.Model):
    
    names = ['M1']
    bounds = [[3,50]]
    
    def __init__(self, events, *args, **kwargs):
    
        super(MassModel,self).__init__()
        
        mean  = (bounds[0][1]+bounds[0][0])/2.
        sigma = (bounds[0][1]-bounds[0][0])/6.
        
        self.events     = events # list of likelihood_DP?
        self.prior      = lk.DP_prior([mean, sigma])# inserire funzione di densità del prior
        
    def log_prior(self, x):
        logP = np.log(self.prior(x['M1']))
        return logP
    
    def log_likelihood(self, x):
        logL = 0.
        for event in self.events:
                logL += event.density(x['M1'])
        return logL

if __name__ == '__main__':
    
    
