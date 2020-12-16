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
    
    def __init__(self, samples, *args, **kwargs):
    
        super(MassModel,self).__init__()
        
        self.samples    = samples # list of lists?
        self.N          = len(self.samples)
        self.prior      = # inserire funzione di densità del prior
        self.likelihood = # inserire funzione di densità della likelihood
        
    def log_prior(self, x):
        logP = np.log(self.prior(x['M1']))
        return logP
    
    def log_likelihood(self, x):
        logL = 0.
        for sample in self.samples:
            logL += self.likelihood(sample)
        
