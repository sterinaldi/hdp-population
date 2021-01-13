import numpy as np
import os
import gibbs_sampler as GS
import ray
from ray.util import ActorPool

events_path = '/Users/stefanorinaldi/Documents/mass_inference/gaussian-40/events/'
event_files = [f for f in os.listdir(events_path) if not f.startswith('.')]
events      = []
output      = '/Users/stefanorinaldi/Documents/mass_inference/gaussian-40'

for event in event_files:
    events.append(np.genfromtxt(events_path+event))

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    


ray.init()
pars = [25,4]
pars_1 = [15, 2]
pars_2 = [35, 3]

samp = []
n_parallels = 4

for i in range(n_parallels):
    samp.append(GS.gibbs_sampler.remote(samples = events,
                        mass_b  = [5,50],
                        n_draws = 1000,
                        burnin  = 1000,
                        step    = 10,
                        alpha0  = 10,
                        gamma   = 10,
                        output_folder = output,
                        verbose = False,
                        # injected_density = lambda x : normal_density(x, *pars)
                        injected_density = lambda x : (normal_density(x, *pars_1) + normal_density(x, *pars_2))/2.
                        ))
                        

pool = ActorPool(samp)
S = pool.map_unordered(lambda a, _ : a.run.remote(), range(4))
for s in S:
    print('success!')
np.savetxt(output+'/mass_samples.txt', np.array([m for s in samp for m in s.mass_samples]))
# producing combined posterior plots
samp[0].postprocessing('/Users/stefanorinaldi/Documents/mass_inference/multivariate-event/mass_samples.txt', bootstrapping = True)
