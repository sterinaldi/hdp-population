import numpy as np
import os
import gibbs_sampler

events_path = 'path/to/events'
event_files = [f for f in os.listdir(samples_path) if not f.startswith('.')]
events      = []
output      = 'path/to/output'

for event in event_files:
    events.append(np.genfromtxt(event))

sampler = gibbs_sampler(samples = events,
                        mass_b  = [5,50],
                        n_draws = 1000,
                        burnin  = 1000,
                        step    = 100,
                        alpha0  = 1,
                        gamma   = 1,
                        output_folder = output
                        )

sampler.run()
