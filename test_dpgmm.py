import numpy as np
import os
import collapsed_gibbs as DPGMM

events_path = '/Users/stefanorinaldi/Documents/mass_inference/universe/events/'
event_files = [f for f in os.listdir(events_path) if not f.startswith('.')]
events      = []
output      = '/Users/stefanorinaldi/Documents/mass_inference/universe/'

for event in event_files:
    events.append(np.genfromtxt(events_path+event))

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)


pars = [25,4]
pars_1 = [25, 3]
pars_2 = [35, 2]

sampler = DPGMM.CGSampler(events = events,
                        #mass_b  = [5,50],
                        n_draws = 100,
                        burnin  = 1000,
                        step    = 100,
                        alpha0  = 2,
                        gamma0   = 1,
                        output_folder = output,
                        # injected_density = lambda x : normal_density(x, *pars)
                        # injected_density = lambda x : (normal_density(x, *pars_1) + normal_density(x, *pars_2))/2.
                        )
                        
sampler.run_event_sampling()
