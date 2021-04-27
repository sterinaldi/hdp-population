import numpy as np
import os
import collapsed_gibbs as DPGMM

events_path = '/Users/stefanorinaldi/Documents/mass_inference/stars_2/events/'
event_files = [f for f in os.listdir(events_path) if not f.startswith('.')]
events      = []
output      = '/Users/stefanorinaldi/Documents/mass_inference/stars_2/'

for event in event_files:
    events.append(np.genfromtxt(events_path+event))

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

sampler = DPGMM.StarClusters(catalog = events[0],
                        n_draws = 100,
                        burnin  = 10,
                        step    = 1,
                        alpha0  = 1,
                        output_folder = output
                        )
                        
sampler.run()
