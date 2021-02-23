import numpy as np
import os
import collapsed_gibbs as DPGMM
import ray
import cpnest.model

events_path = '/Users/stefanorinaldi/Documents/mass_inference/multidim/events/'
event_files = [f for f in os.listdir(events_path) if not f.startswith('.')]
events      = []
output      = '/Users/stefanorinaldi/Documents/mass_inference/multidim/'

for event in event_files:
    events.append(np.genfromtxt(events_path+event))

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

sampler = DPGMM.CGSampler(events = events,
                        n_draws = 10,
                        burnin  = 100,
                        step    = 10,
                        alpha0  = 1,
                        output_folder = output,
                        # injected_density = lambda x : normal_density(x, *pars)
                        # injected_density = lambda x : (normal_density(x, *pars_1) + normal_density(x, *pars_2))/2.
                        )
                        
sampler.run_event_sampling()
