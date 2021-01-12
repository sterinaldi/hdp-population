import numpy as np
import os
import gibbs_sampler as GS

events_path = '/Users/stefanorinaldi/Documents/mass_inference/multivariate-25-35/events/'
event_files = [f for f in os.listdir(events_path) if not f.startswith('.')]
events      = []
output      = '/Users/stefanorinaldi/Documents/mass_inference/multivariate-25-35'

for event in event_files:
    events.append(np.genfromtxt(events_path+event))

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

pars = [40,3]
pars_1 = [25, 3]
pars_2 = [35, 2]

sampler = GS.gibbs_sampler(samples = events,
                        mass_b  = [5,50],
                        n_draws = 1000,
                        burnin  = 1000,
                        step    = 10,
                        alpha0  = 10,
                        gamma   = 10,
                        output_folder = output,
                        #injected_density = lambda x : normal_density(x, *pars)
                        injected_density = lambda x : (normal_density(x, *pars_1)+ normal_density(x, *pars_2))/2.
                        )

sampler.run()
#sampler.postprocessing('/Users/stefanorinaldi/Documents/mass_inference/mass_samples.txt', bootstrapping = True)
