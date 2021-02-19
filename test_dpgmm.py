import numpy as np
import os
import collapsed_gibbs as DPGMM

events_path = '/Users/stefanorinaldi/Documents/mass_inference/universe_1/events/'
event_files = [f for f in os.listdir(events_path) if not f.startswith('.')]
events      = []
output      = '/Users/stefanorinaldi/Documents/mass_inference/universe_1/'

for event in event_files:
    events.append(np.genfromtxt(events_path+event))

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)


pars = [30,3]
pars_1 = [30, 3]
pars_2 = [35, 2]

sampler = DPGMM.CGSampler(events = events,
                        #mass_b  = [5,50],
                        samp_settings = [100,10,100],
                        alpha0  = 1,
                        gamma0   = 3,
                        delta_M = 2,
                        output_folder = output,
                        process_events = False,
                        injected_density = lambda x : normal_density(x, *pars)
                        # injected_density = lambda x : (normal_density(x, *pars_1) + normal_density(x, *pars_2))/2.
                        )
                        
sampler.run()
