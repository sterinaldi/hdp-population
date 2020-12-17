import numpy as np
from numpy.random import rand, normal
from random import choice

def Gaussian(x0, mu, sigma):
    return np.exp((x0-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

class prior_DP:
    
    def __init__(self,
                pars,
                G0 = None,
                alpha = 1
                ):
        self.pars    = pars
        self.pts     = []
        self.alpha   = alpha
        self.n_pts   = 0
        if G0 is not None:
            self.G0 = G0
        else:
            self.G0 = Gaussian # a simple choice
    
    def density(self, x):
        prob = self.alpha/(self.alpha + self.n_pts) * self.G0(x,*self.pars)
        if x in self.pts:
            prob += self.pts.count(x)/(self.alpha + self.n_pts)
        self.pts.append(x)
        return prob
        
class likelihood_DP:
    
    def __init__(self,
                samples,
                pars,
                G = None,
                alpha = 1
                ):
        self.pars      = pars
        self.samples   = samples
        self.alpha     = alpha
        self.n_samples = len(samples)
        if G is not None:
            self.G = G
        else:
            self.G = Gaussian # a simple choice

    def density(self, x):
        prob = self.alpha/(self.alpha + self.n_samples) * self.G(x,*self.pars)
        if x in self.pts:
            prob += self.pts.count(x)/(self.alpha + self.n_samples)
        return prob
