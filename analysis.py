import numpy as np
import os
import gibbs_sampler as GS
import ray

events_path = '/Users/stefanorinaldi/Documents/mass_inference/multivariate-event/events/'
event_files = [f for f in os.listdir(events_path) if not f.startswith('.')]
events      = []
output      = '/Users/stefanorinaldi/Documents/mass_inference/multivariate-event'

for event in event_files:
    events.append(np.genfromtxt(events_path+event))

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

@ray.remote
def wrapper(sampler):
    sampler.run()
    return sampler.get_mass_samples()

ray.init()
pars = [25,4]
pars_1 = [15, 2]
pars_2 = [35, 3]

samplers = []
n_parallel_jobs = 4

for i in range(n_parallel_jobs):
    samplers.append(GS.gibbs_sampler(samples = events,
                        mass_b  = [5,50],
                        n_draws = 10,
                        burnin  = 100,
                        step    = 10,
                        alpha0  = 10,
                        gamma   = 10,
                        output_folder = output,
                        verbose = True,
                        # injected_density = lambda x : normal_density(x, *pars)
                        injected_density = lambda x : (normal_density(x, *pars_1) + normal_density(x, *pars_2))/2.
                        ))
                        
tasks = []
for sampler in samplers:
    tasks.append(wrapper.remote(sampler))
mass_samples = [m for list in ray.get(tasks) for m in list]
np.savetxt(output+'/mass_samples.txt', np.array([mass_samples]))
# producing combined posterior plots
samplers[0].postprocessing(output+'/mass_samples.txt', bootstrapping = True)
