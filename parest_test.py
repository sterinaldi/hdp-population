import numpy as np
import matplotlib.pyplot as plt
from ParEst import DirichletDistribution
import pickle
import cpnest
import corner
from scipy.special import erf
from scipy.stats import norm, dirichlet
from scipy.interpolate import interp1d
import os

def PL(m, alpha, m_max, m_min, l_max, l_min):
    f = m**(-alpha)*(1+erf((m-m_min)/(l_min)))*(1+erf((m_max-m)/l_max))/4.
    return f

def gauss(x, x0, s):
    return np.exp(-((x-x0)**2/(2*s**2)))/(np.sqrt(2*np.pi)*s)

def logPrior(mu, sigma):
    return 0
 
#true_vals = [1.2, 75, 20, 10, 5, 1]
mu = 25
s = 3
a = 100
true_vals = [mu, s, a]

#samp_file = '/Users/stefanorinaldi/Documents/mass_inference/PL_peak/reconstructed_events/pickle/posterior_functions_event_1.pkl'
#openfile  = open(samp_file, 'rb')
#samples   = pickle.load(openfile)
#openfile.close()

x = np.linspace(20,30,20)
dx = x[1]-x[0]
ps = norm(loc = mu, scale = s).pdf(x)

samples  = dirichlet(a*ps*dx/np.sum(ps*dx)).rvs(size = 30)

out_folder  = '/Users/stefanorinaldi/Documents/mass_inference/PL_no_peak/PE'

fig = plt.figure()
ax  = fig.add_subplot(111)
for s in samples:
    ax.plot(x, s, linewidth = 0.3)
ax.plot(x,ps*dx/np.sum(ps*dx))

ax.set_xlabel('x')
ax.set_ylabel('p(x)dx')
fig.suptitle('Conc. par. = {0}'.format(a))
plt.savefig(out_folder+'/draws.pdf', bbox_inches = 'tight')

names  = ['mu', 'sigma']#['alpha', 'm_max', 'm_min', 'l_max', 'l_min']
bounds = [[23, 28], [2,4]]#[[0,2], [70,80], [15,30], [8,12], [3,7]]
labels  = ['\\mu', '\\sigma']#['\\alpha', 'm_{max}', 'm_{min}', '\\lambda_{max}', '\\lambda_{min}']

PE = DirichletDistribution(
    gauss,
    names,
    bounds,
    samples,
    min(x),
    max(x),
    prior_pars = logPrior,
    probs = samples,
    n = len(samples[0])
    )

work = cpnest.CPNest(PE,
                    verbose = 2,
                    nlive = 1000,
                    maxmcmc = 5000,
                    nthreads = 2,
                    output  = out_folder
                    )
work.run()
print('log Evidence: {0}'.format(work.NS.logZ))

labels = labels + ['a']
names = names + ['a']
x = work.posterior_samples.ravel()
samps = np.column_stack([x[lab] for lab in names])
fig = corner.corner(samps,
       labels= [r'${0}$'.format(lab) for lab in labels],
       quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
       show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
       use_math_text=True, truths=true_vals,
       filename=os.path.join(out_folder,'joint_posterior.pdf'))
fig.savefig(os.path.join(out_folder,'joint_posterior.pdf'), bbox_inches='tight')
