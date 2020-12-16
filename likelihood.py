import numpy as np
from numpy.random import rand, normal
from random import choice

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
            self.G0 = normal # a simple choice
    
    def density(self, x):
        prob = self.alpha/(self.alpha + self.n_pts) * G0(x,*pars)
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
        self.pars    = pars
        self.samples =
        self.alpha   = alpha
        self.n_pts   = 0
        if G is not None:
            self.G = G
        else:
            self.G = normal # a simple choice

    def density(self, x):
        prob = self.alpha/(self.alpha + self.n_pts) * G(x,*pars)
        if x in self.pts:
            prob += self.pts.count(x)/(self.alpha + self.n_pts)
        return prob
