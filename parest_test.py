import numpy as np
import matplotlib.pyplot as plt
from ParEst import DirichletDistribution, DirichletProcess
import pickle
import cpnest
import corner
from scipy.special import erf, logsumexp
from scipy.stats import norm, dirichlet
from scipy.interpolate import interp1d
import os


def log_add(x, y): return x+np.log(1.0+np.exp(y-x)) if x >= y else y+np.log(1.0+np.exp(x-y))
def log_norm(x, x0, s): return -((x-x0)**2)/(2*s*s) - np.log(np.sqrt(2*np.pi)) - np.log(s)

def PL(m, alpha, m_max, m_min, l_max, l_min):
    f = m**(-alpha)*(1+erf((m-m_min)/(l_min)))*(1+erf((m_max-m)/l_max))/4.
    return f

def gauss(x, x0, s):
    return np.exp(-((x-x0)**2/(2*s**2)))/(np.sqrt(2*np.pi)*s)

def bimodal(x,x0,s0,x1,s1):
    return (np.exp(-((x-x0)**2/(2*s0**2)))/(np.sqrt(2*np.pi)*s0) + np.exp(-((x-x1)**2/(2*s1**2)))/(np.sqrt(2*np.pi)*s1))/2.

def logPrior(*args):
    return 0
 
#
alpha = 1.1
mmax = 75
mmin = 15
lmax = 5
lmin = 5
mu = 30
sigma = 6
a = 20
#true_vals = [alpha, mmax, mmin, lmax, lmin, a, 1]
true_vals = [mu, sigma, a, 1]
#true_vals = [50, 2, 100, 2.5, a, 1]

samp_file = '/Users/stefanorinaldi/Documents/mass_inference/DP/reconstructed_events/pickle/posterior_functions_event_0.pkl'
openfile  = open(samp_file, 'rb')
samples   = pickle.load(openfile)
openfile.close()


x = np.linspace(10, 50,300)
dx = x[1]-x[0]

##
#ps = PL(x, alpha, mmax, mmin, lmax, lmin)
#
#samples  = dirichlet(a*ps*dx/np.sum(ps*dx)).rvs(size = 300)
#
out_folder  = '/Users/stefanorinaldi/Documents/parametric/DP'
#
#samples = np.array([s for s in samples if (s != 0.).all()])
#
#fig = plt.figure()
#ax  = fig.add_subplot(111)
#for s in samples:
#    ax.plot(x, s, linewidth = 0.3)
#
#ax.plot(x,ps*dx/np.sum(ps*dx))
#
#ax.set_xlabel('x')
#ax.set_ylabel('p(x)dx')
#fig.suptitle('Conc. par. = {0}'.format(a))
#plt.savefig(out_folder+'/draws.pdf', bbox_inches = 'tight')

#probs = []
#for samp in samples:
#    p = np.ones(300) * -np.inf
#    for component in samp.values():
#        logW = np.log(component['weight'])
#        mu   = component['mean']
#        s    = component['sigma']
#        for i, mi in enumerate(x):
#            p[i] = log_add(p[i], logW + log_norm(mi, mu, s))
#    p = np.exp(p + np.log(dx) - logsumexp(p+np.log(dx)))
#    probs.append(p)
#
#for p in probs:
#    plt.plot(x,p, lw = 0.3)
#
#plt.plot(x, gauss(x, 30.075, 5.945)*dx/np.sum(gauss(x, 30.075, 5.945)*dx), c = 'r')
#plt.show()

#names  = ['alpha', 'm_max', 'm_min', 'l_max', 'l_min']
#bounds = [[0,2], [60,80], [10,30], [3,10], [3,10]]
#labels  = ['\\alpha', 'm_{max}', 'm_{min}', '\\lambda_{max}', '\\lambda_{min}']
#
#names = ['mu_1', 'sigma_1', 'mu_2', 'sigma_2']
#bounds = [[40,60], [1,4], [90, 110], [1,4]]
#labels = ['\mu_1', '\sigma_1', '\mu_2', '\sigma_2']
names = ['mu', 'sigma']
bounds = [[20,45], [3,7]]
labels = ['\mu', '\sigma']


PE = DirichletProcess(
    gauss,
    names,
    bounds,
    samples[:300],
    10,
    50,
    prior_pars = logPrior,
    max_a = 10000,
    max_N = 200
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

labels = labels + ['a', 'N']
names = names + ['a','N']
x = work.posterior_samples.ravel()
samps = np.column_stack([x[lab] for lab in names])
fig = corner.corner(samps,
       labels= [r'${0}$'.format(lab) for lab in labels],
       quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
       show_titles=True, title_fmt='.3f', title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 16},
       use_math_text=True, truths=true_vals,
       filename=os.path.join(out_folder,'joint_posterior.pdf'))
fig.savefig(os.path.join(out_folder,'joint_posterior.pdf'), bbox_inches='tight')
