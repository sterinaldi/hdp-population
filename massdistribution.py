import numpy as np
from dpgmm.dpgmm import * # rivedere dopo installazione
import multiprocessing as mp
import matplotlib
import dill as pickle
import os
import matplotlib.pyplot as plt
from scipy.special import logsumexp

class DPGMM_mass(object):
    '''
    DPGMM class for mass function.
    
    
    '''

    def __init__(self,
                mass_samples,  # da cambiare per introdurre posteriors
                max_sticks = 10,
                bins = 10,
                output = './',
                M_min = 3,
                M_max = 50,
                gridpts = 1000
                ):
    
        self.mass_samples  = np.array(mass_samples)
        self.dims               = 1
        self.max_sticks         = max_sticks
        self.nthreads           = mp.cpu_count()//2
        self.bins               = bins
        self.pool               = mp.Pool(self.nthreads)
        self.output             = output
        self.grid               = np.linspace(M_min, M_max, gridpts)
    
    def _initialise_dpgmm(self):
        self.model = DPGMM(self.dims)
        for sample in self.mass_samples:
            self.model.add([sample])
        
        self.model.setPrior(mean = np.mean(self.mass_samples))
        self.model.setThreshold(1e-4)
        self.model.setConcGamma(1,1)
    
    def compute_dpgmm(self):
        self._initialise_dpgmm()
        
        solve_args = [(nc, self.model) for nc in range(1,self.max_sticks+1)]
        solve_results = self.pool.map(solve_dpgmm, solve_args)
        self.scores = np.array([r[1] for r in solve_results])
        self.model = (solve_results[self.scores.argmax()][-1])
        pickle.dump(self.model, open(os.path.join(self.output,'dpgmm_model.p'), 'wb'))
        print("best model has ",self.scores.argmax()+1,"components")
        self.density = self.model.intMixture()
        pickle.dump(self.density, open(os.path.join(self.output,'dpgmm_density.p'), 'wb'))
        
        self.density_array = np.array([Posterior((self.density, point)) for point in self.grid])
    
    def display_density(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.mass_samples, bins = int(np.sqrt(len(self.mass_samples))), density = True)
        ax.plot(self.grid, self.density_array)
        plt.show()
    

def solve_dpgmm(args):
    (nc, model_in) = args
    model          = DPGMM(model_in)
    for _ in range(nc-1): model.incStickCap()
#    try:
    it = model.solve(iterCap=1024)
    return (model.stickCap, model.nllData(), model)

def Posterior(args):
    density, mass = args
    logPs = [np.log(density[0][ind])+prob.logProb(mass) for ind,prob in enumerate(density[1])]
    return np.exp(logsumexp(logPs))
