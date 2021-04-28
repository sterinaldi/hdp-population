import numpy as np
import os
from numpy.random import normal, uniform, choice
import sampler
import matplotlib.pyplot as plt

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def draw_sample(mu1, mu2, s1, s2):
    while(1):
        m = uniform(5, 50)
        if (normal_density(m, mu1, s1) + normal_density(m, mu2, s2)) > uniform(0, max([normal_density(mu1,mu1,s1), normal_density(mu2,mu2,s2)])):
            return m

out_dir   = '/Users/stefanorinaldi/Documents/mass_inference/test_generator_2/events/'
out_plot  = '/Users/stefanorinaldi/Documents/mass_inference/test_generator_2/'
n_events  = 1
n_samples = 200
alpha     = 1000
pars      = [25,4]
samples = []

if not os.path.exists(out_plot):
    os.mkdir(out_plot)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

mu = []
for i in range(n_events):
    #pars = [draw_sample(25, 35, 3, 2), np.exp(uniform(np.log(2), np.log(6)))]
    pars  = [normal(10, 3), normal(35,2),np.exp(uniform(np.log(2), np.log(6))), np.exp(uniform(np.log(2), np.log(6)))]
    for _ in range(n_samples):
        if uniform() < alpha/(alpha+len(samples)):
            samples.append(draw_sample(*pars))
            #samples.append(normal(*pars))
        else:
            samples.append(choice(samples))
    mu.append(pars[0])
    np.savetxt(out_dir+'event_{0}.txt'.format(i+1), np.array(samples))
    samples = []

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(mu, bins = int(np.sqrt(len(mu))))
ax.set_xlabel('$M_1\ [M_\\odot]$')
fig.savefig(out_plot + '/M1_events.pdf', bbox_inches = 'tight')

np.savetxt(out_plot + '/M1_events.txt', np.array(mu))