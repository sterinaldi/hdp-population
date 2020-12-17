import numpy as np
from numpy.random import normal

out_dir   = '/Users/stefanorinaldi/Documents/mass_inference/events/'
n_events  = 100
n_samples = 1000

for i in range(n_events):
    M1 = normal(30,7)
    file = out_dir + 'event_%d.txt' %(i+1)
    np.savetxt(file, normal(M1, 3, n_samples))

