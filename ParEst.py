import cpnest.model
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import gamma, dirichlet
import numpy as np
from numba.extending import get_cython_function_address
from numba import vectorize, njit
import ctypes

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

def log_add(x, y): return x+np.log(1.0+np.exp(y-x)) if x >= y else y+np.log(1.0+np.exp(x-y))
def log_norm(x, x0, s): return -((x-x0)**2)/(2*s*s) - np.log(np.sqrt(2*np.pi)) - np.log(s)

class DirichletDistribution(cpnest.model.Model):
    
    def __init__(self, model, pars, bounds, samples, x_min, x_max, n = 30, prior_pars = lambda x: 0, probs = None):
    
        super(DirichletDistribution, self).__init__()
        self.samples    = samples
        self.names      = pars + ['a']
        self.bounds     = bounds + [[0, 200]]
        self.prior_pars = prior_pars
        self.x_min      = x_min
        self.x_max      = x_max
        self.n          = n
        self.m          = np.linspace(self.x_min, self.x_max, self.n)
        self.dm         = self.m[1] - self.m[0]
        self.model      = model
        
        if probs is None:
            for samp in self.samples:
                p = np.ones(self.n) * -np.inf
                for component in samp.values():
                    logW = np.log(component['weight'])
                    mu   = component['mean']
                    s    = component['sigma']
                    for i, mi in enumerate(self.m):
                        p[i] = log_add(p[i], logW + log_norm(mi, mu, s))
                    p = p + np.log(self.dm) - logsumexp(p+np.log(self.dm))
                probs.append(p)
        self.probs = np.log(np.array(probs))
    
    def log_prior(self, x):
    
        logP = super(DirichletDistribution,self).log_prior(x)
        if np.isfinite(logP):
            logP = gamma(1,1).logpdf(x['a'])
            pars = [x[lab] for lab in self.names[:-1]]
            logP += self.prior_pars(*pars)
        return logP

    def log_likelihood(self, x):

        pars = [x[lab] for lab in self.names if not lab == 'a']
        base = np.array([self.model(mi, *pars)*self.dm for mi in self.m])
        base = base/np.sum(base)
        a = x['a']*base
#        logL = np.sum([dirichlet(a).logpdf(p) for p in self.probs])
        logL = self.n * (numba_gammaln(np.sum(a)) - np.sum([numba_gammaln(ai) for ai in a]))
        for p in self.probs:
            logL += np.sum((a - 1)*p) #scipy.stats.dirichlet implementation w/o checks
        
        return logL

@njit
def numba_gammaln(x):
  return gammaln_float64(x)
