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
    
    def __init__(self, model, pars, bounds, samples, x_min, x_max, probs, n = 30, prior_pars = lambda x: 0, max_a = 2000):
    
        super(DirichletDistribution, self).__init__()
        self.samples    = samples
        self.labels     = pars
        self.names      = pars + ['a']
        self.bounds     = bounds + [[0, max_a]]
        self.prior_pars = prior_pars
        self.x_min      = x_min
        self.x_max      = x_max
        self.n          = n
        self.m          = np.linspace(self.x_min, self.x_max, self.n)
        self.dm         = self.m[1] - self.m[0]
        self.model      = model
        self.probs      = np.array(probs)
    
    def log_prior(self, x):
    
        logP = super(DirichletDistribution,self).log_prior(x)
        if np.isfinite(logP):
            logP = 0.
            pars = [x[lab] for lab in self.labels]
            logP += self.prior_pars(*pars)
        return logP

    def log_likelihood(self, x):

        pars = [x[lab] for lab in self.labels]
        base = np.array([self.model(mi, *pars)*self.dm for mi in self.m])
        base = base/np.sum(base)
        a = x['a']*base
        #implemented as in scipy.stats.dirichlet.logpdf() w/o checks
        lnB = np.sum([numba_gammaln(ai) for ai in a]) - numba_gammaln(np.sum(a))
        logL = np.sum([- lnB + np.sum((xlogy(a-1, p.T)).T, 0) for p in self.probs])
        return logL

class DirichletProcess(cpnest.model.Model):
    
    def __init__(self, model, pars, bounds, samples, x_min, x_max, n = 30, prior_pars = lambda x: 0, max_a = 2000, max_N = 1e4):
    
        super(DirichletDistribution, self).__init__()
        self.samples    = samples
        self.labels     = pars
        self.names      = pars + ['a', 'N']
        self.bounds     = bounds + [[0, max_a], [1,max_N]]
        self.prior_pars = prior_pars
        self.x_min      = x_min
        self.x_max      = x_max
        self.model      = model
        self.prec_probs = {}

    
    def log_prior(self, x):
    
        logP = super(DirichletDistribution,self).log_prior(x)
        if np.isfinite(logP):
            logP = -np.log(x['N'])
            pars = [x[lab] for lab in self.labels]
            logP += self.prior_pars(*pars)
        return logP

    def log_likelihood(self, x):
        N = int(x['N'])
        self.m          = np.linspace(self.x_min, self.x_max, N)
        self.dm         = self.m[1] - self.m[0]
        if N in self.prec_probs.keys():
            probs = self.prec_probs[N]
        else:
            for samp in self.samples:
                p = np.ones(N) * -np.inf
                for component in samp.values():
                    logW = np.log(component['weight'])
                    mu   = component['mean']
                    s    = component['sigma']
                    for i, mi in enumerate(self.m):
                        p[i] = log_add(p[i], logW + log_norm(mi, mu, s))
                    p = np.exp(p + np.log(self.dm) - logsumexp(p+np.log(self.dm)))
                probs.append(p)
            self.prec_probs[N] = probs
        
        pars = [x[lab] for lab in self.labels]
        base = np.array([self.model(mi, *pars)*self.dm for mi in self.m])
        base = base/np.sum(base)
        a = x['a']*base
        #implemented as in scipy.stats.dirichlet.logpdf() w/o checks
        lnB = np.sum([numba_gammaln(ai) for ai in a]) - numba_gammaln(np.sum(a))
        logL = np.sum([- lnB + np.sum((xlogy(a-1, p.T)).T, 0) for p in probs])
        return logL


@njit
def numba_gammaln(x):
  return gammaln_float64(x)
