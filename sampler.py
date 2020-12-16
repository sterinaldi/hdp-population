import numpy as np
from numpy.random import rand, normal
from random import choice
import matplotlib.pyplot as plt

class posterior_DP:
    
    def __init__(self,
                pars, # iterable
                G = None, # base distribution
                alpha = 1. # concentration parameter
                ):
        self.pars = pars
        self.samples = []
        self.alpha = alpha
        self.n_draws = 0
        if G is not None:
            self.G = G
        else:
            self.G = normal
        
    def generate_samples(self, nsamples):
    
        for _ in range(nsamples):
            self.draw_sample()
        return np.array(self.samples)
    
    def draw_sample(self):
        if rand() < self.alpha/(self.alpha+self.n_draws):
            self.samples.append(self.G(*self.pars))
        else:
            self.samples.append(choice(self.samples))
        self.n_draws += 1
        
    def initialise(self):
        self.samples = []
        self.n_draws = 0
    
class mass_DP:

    def __init__(self,
                pars,
                G0 = None,
                alpha=1.
                ):
        self.pars = pars
        self.pars_samples = []
        self.mass_samples = []
        self.alpha = alpha
        self.n_draws = 0
        if G0 is not None:
            self.G0 = G0
        else:
            self.G0 = normal
    
    def generate_samples(self, nsamples, n_samples_post = 100, alpha = 1, G = None):
        for _ in range(nsamples):
            pars = self.draw_pars()
            event = posterior_DP(pars, G = G, alpha = alpha)
            self.pars_samples.append(pars)
            self.mass_samples.append(event.generate_samples(n_samples_post))
        return self.pars_samples
    
    def draw_pars(self):
        if rand() < self.alpha/(self.alpha+self.n_draws):
            self.n_draws += 1
            return [self.G0(*self.pars)]
        else:
            self.n_draws += 1
            return choice(self.pars_samples)

        
    def initialise(self):
        self.mass_samples = []
        self.pars_samples = []
        self.n_draws = 0
    
    def plot_samples(self, support = [0,50]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(np.array(self.pars_samples)[:,0], bins = 'auto', density = True)
        ax.set_xlabel('$M_1 [M_\\odot]$')
        
        
