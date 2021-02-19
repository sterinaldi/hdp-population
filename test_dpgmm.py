import numpy as np
import os
import collapsed_gibbs as DPGMM

events_path = '/Users/stefanorinaldi/Documents/mass_inference/multivariate-25-35/events/'
event_files = [f for f in os.listdir(events_path) if not f.startswith('.')]
events      = []
output      = '/Users/stefanorinaldi/Documents/mass_inference/multivariate-25-35/'

for event in event_files:
    events.append(np.genfromtxt(events_path+event))

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)


pars = [25,4]
pars_1 = [25, 3]
pars_2 = [35, 2]

sampler = DPGMM.CGSampler(events = events,
                        #mass_b  = [5,50],
                        samp_settings = [100,10,10],
                        alpha0  = 1,
                        gamma0   = 1,
                        output_folder = output,
                        process_events = True
                        # injected_density = lambda x : normal_density(x, *pars)
                        # injected_density = lambda x : (normal_density(x, *pars_1) + normal_density(x, *pars_2))/2.
                        )
                        
sampler.run()
