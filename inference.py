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
import configparser
from cpnest import nest2pos
import numpy as np

class MassModel(cpnest.model.Model):
    
    def __init__(self, events, *args, **kwargs):
    
        super(MassModel,self).__init__()
        
        self.names = ['M1']
        self.bounds = [[3,50]]
        mean  = (self.bounds[0][1]+self.bounds[0][0])/2.
        sigma = (self.bounds[0][1]-self.bounds[0][0])/6.
        
        self.events     = events # list of likelihood_DP?
        self.prior      = lk.prior_DP([mean, sigma])# inserire funzione di densità del prior
        
    def log_prior(self, x):
        logP = np.log(self.prior.density(x['M1']))
        return logP
    
    def log_likelihood(self, x):
        logL = 0.
        for event in self.events:
                logL += np.log(event.density(x['M1']))
        return logL

if __name__ == '__main__':
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    opts   = config['DEFAULT']
    output = opts['output']
    if opts['threads'] == 'None':
        nthreads = None
    else:
        nthreads = int(opts['threads'])
    
    events = []
    events_list = os.listdir(opts['events_dir'])
    for event in events_list:
        samples = np.genfromtxt(opts['events_dir']+event)
        events.append(lk.likelihood_DP(samples, [], lambda x: 1))
    
    M = MassModel(events)
    
    work = cpnest.CPNest(M,
                        verbose      = int(opts['verbose']),
                        poolsize     = int(opts['poolsize']),
                        nthreads     = nthreads,
                        nlive        = int(opts['nlive']),
                        maxmcmc      = int(opts['maxmcmc']),
                        output       = output,
                        nhamiltonian = 0)
    work.run()
    print('log Evidence {0]'.format(work.NS.logZ))
    
    # output
    x = work.posterior_samples.ravel()
    samps = np.column_stack((x['h']))
    fig = corner.corner(samps,
                        labels = [r'$M_1\ [M_\\odot]$'],
                        show_titles = True,
                        title_kwargs={"fontsize": 12},
                        use_math_text=True,
                        filename=os.path.join(output,'joint_posterior.pdf')
                        )
    fig.savefig(os.path.join(output,'joint_posterior.pdf'), bbox_inches='tight')
    
    
