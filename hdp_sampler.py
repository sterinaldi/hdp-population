import numpy as np
from numpy.random import uniform
import numpy.random as rd
import matplotlib.pyplot as plt
import os
import corner
from numba import jit
from numba.experimental import jitclass
from numba import types, typed
from scipy.special import logsumexp

@jit
def draw_mass(Mold, delta_M):
    return Mold + delta_M * uniform(-1,1)

@jit
def draw_sigma(old_sigma, delta_s):
    return np.exp(np.log(old_sigma) + delta_s * uniform(-1,1))

@jit
def draw_alpha(old_alpha, delta_a):
    return old_alpha + delta_a * uniform(-1,1)

class hdp_sampler:
    
    """
    MCMC Gibbs sampler: Hierarchical Dirichlet Process.

    Based on Teh et al. (2006), sec. 5.1, follows the same notation.
    https://www.researchgate.net/publication/4742259_Hierarchical_Dirichlet_Processes
    ------------------
    """
    
    def __init__(self,
                 events,
                 mass_b,
                 n_draws,
                 burnin,
                 step,
                 gamma,
                 sigma_b = [np.log(1), np.log(10)],
                 alpha_b = [1, 50],
                 output_folder = './',
                 delta_M = 5,
                 delta_s = 0.15,
                 delta_a = 5,
                 injected_density = None,
                 verbose = True,
                 diagnostic = False,
                 truths = None):
    
        self.events      = events
        
        self.max_m     = max(mass_b)
        self.min_m     = min(mass_b)
        self.max_sigma = max(sigma_b)
        self.min_sigma = min(sigma_b)
        self.max_alpha = max(alpha_b)
        self.min_alpha = min(alpha_b)
        
        # Uniform prior on samples
        self.samples_prior = lambda : -np.log(self.max_m - self.min_m) #if (self.min_m < x < self.max_m) else 0
        
        # Configuration parameters
        self.gamma       = gamma       # Outer DP concentration parameter
        self.n_draws     = n_draws     # total number of outcomes
        self.burnin      = burnin      # burn-in
        self.step        = step        # steps between two outcomes (avoids autocorrelation)
        self.delta_M     = delta_M     # interval around old M sample (for updating)
        self.delta_s     = delta_s     # interval around old sigma sample (for updating)
        self.delta_a     = delta_a     # interval around old alpha sample (for updating)
        
        self.output_folder    = output_folder
        self.injected_density = injected_density
        self.verbose          = verbose
        self.diagnostic       = diagnostic
        self.truths           = truths
        
        self.mass_samples  = np.zeros(self.n_draws * len(self.events))
        self.sigma_samples = np.zeros(self.n_draws * len(self.events))
        self.alpha_samples = np.zeros(self.n_draws * len(self.events))
        
        self.single_event_samples = np.array([np.array([np.zeros(3) for _ in range(n_draws)]) for _ in range(len(events))])
        
        self.acceptance = []
        self.accept     = 0.
        
        self.initialise_pars()
        
        return
        
    def initialise_pars(self):
        """
        Initialise a random set of parameters (mu, sigma, alpha) for each event
        """
        
        self.pars = np.array([np.array([uniform(self.min_m, self.max_m), np.exp(uniform(self.min_sigma, self.max_sigma)), uniform(self.min_alpha, self.max_alpha)]) for _ in range(len(self.events))])

    def update_pars(self, event_index):
        """
        Updates the parameters for the local DP
        _____________
        Arguments:
            :int event_index: Event index.
        """
        
        old_pars = self.pars[event_index]
        
        if uniform() < self.gamma/(self.gamma+len(self.pars)):
            new_pars = np.array([draw_mass(old_pars[0], self.delta_M),
                                 draw_sigma(old_pars[1], self.delta_s),
                                 draw_alpha(old_pars[2], self.delta_a)])
        else:
            new_i = rd.randint(len(self.events))
            new_pars = self.pars[new_i]
            
        p_old = self.evaluate_probability_pars(old_pars, event_index)
        p_new = self.evaluate_probability_pars(new_pars, event_index)
        
        if p_new - p_old > np.log(uniform()):
            self.accept += 1.
            self.pars[event_index] = new_pars
        return
    
    def evaluate_probability_pars(self, pars, event_index):
        """
        Evaluates conditional distribution for a specific set of parameters [mu, sigma, alpha]
        Eq. 32 in Teh et al. (2006)
        _____________
        Arguments:
            :iterable pars: List of parameters.
        Returns:
            :double:        Log probability.
        """

        if not (self.min_m < pars[0] < self.max_m and self.min_sigma < np.log(pars[1]) < self.max_sigma and self.min_alpha < pars[2] < self.max_alpha):
            return -np.inf
        
        logP = 0.
        for index in range(len(self.events[event_index])):
            draws = self.events[event_index][:index]
            new   = self.events[event_index][index]
            
            if new in draws:
                logP += np.log(len(draws)/(pars[2] + len(draws))) + np.log(np.sum(draws == new))
                
            else:
                logP += np.log(pars[2]/(len(draws) + pars[2])) + self.log_normal_density(new, pars[0], pars[1])
        
        return logP
    
    def log_normal_density(self, x, x0, sigma):
        """
        Normal probability density function.
        
        ------------
        Arguments:
            :double x:     Point.
            :double x0:    Mean.
            :double sigma: Variance.
        Returns:
            :double:       log(N(x)).
        """
        return (-(x-x0)**2/(2*sigma**2))-np.log(np.sqrt(2*np.pi)*sigma)
    
    def markov_step(self):
        """
        Evolves the Markov Chain and computes acceptance
        """
        
        for event_index in range(len(self.events)):
            self.update_pars(event_index)
        return
    
    def save_posterior_samples(self, draw):
        """
        Stores samples in apposite variable.
        """
        for parameters, index in zip(self.pars, range(len(self.pars))):
            self.mass_samples[draw + index]  = parameters[0]
            self.sigma_samples[draw + index] = parameters[1]
            self.alpha_samples[draw + index] = parameters[2]
            self.single_event_samples[index][draw] = parameters
        return
    
    def run_sampling(self):
        """
        Run sampler - n_draws points.
        """
        
        for i in range(self.burnin):
            self.markov_step()
            if self.verbose:
                print('\rBURN-IN: {0}/{1}'.format(i+1, self.burnin), end = '')
        if self.verbose:
            print('\n', end = '')
        for i in range(self.n_draws):
            self.accept = 0.
            if self.verbose:
                print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for j in range(self.step):
                self.markov_step()
            self.acceptance.append(self.accept/self.step)
            self.save_posterior_samples(i)
        if self.verbose:
            print('\n', end = '')
        return

    def display_config(self):
        print('MCMC Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.events)))
        print('Mass interval: {0}-{1} Msun'.format(self.min_m, self.max_m))
        print('Sigma interval: {0}-{1} Msun'.format(np.exp(self.min_sigma), np.exp(self.max_sigma)))
        print('Alpha interval: {0}-{1}'.format(self.min_alpha, self.max_alpha))
        print('Concentration parameter: gamma = {0}'.format(self.gamma))
        print('Burn-in: {0} samples'.format(self.burnin))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws, self.step))
        print('------------------------')
        return

    def plot_samples(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution.
        """
        
        app  = np.linspace(self.min_m, self.max_m, 1000)
        percentiles = [5,16, 50, 84, 95]
        
        for samples, post, i in zip(self.events, self.single_event_samples, range(len(self.events))):
            set_samples = np.array(list(set(samples)))
            #app = np.sort(np.concatenate((app, set_samples)))
            fig = plt.figure()
            fig.suptitle('Event {0}'.format(i+1))
            ax  = fig.add_subplot(111)
            ax.hist(set(samples), bins = int(np.sqrt(len(set(samples)))), histtype = 'step', density = True)
            self.probs = []
            self.p = {}
            for a in app:
                n = np.sum(samples == a)
                if n == 0:
                    self.probs.append([np.log(sample[2]) + self.log_normal_density(a, sample[0], sample[1]) - np.log(sample[2]) for sample in post])
                else:
                    self.probs.append([logsumexp([np.log(n), np.log(sample[2]) + self.log_normal_density(a, sample[0], sample[1]) - np.log(len(samples) + sample[2])]) for sample in post])
            for perc in percentiles:
                self.p[perc] = np.exp(np.percentile(self.probs, perc, axis = 1))
            np.savetxt(self.output_events + '/rec_prob_{0}.txt'.format(i+1), np.array(self.p[50]))
            ax.fill_between(app, self.p[95], self.p[5], color = 'lightgreen', alpha = 0.5)
            ax.fill_between(app, self.p[84], self.p[16], color = 'aqua', alpha = 0.5)
            ax.plot(app, self.p[50], marker = '', color = 'r')
            ax.set_xlabel('$M_1\ [M_\\odot]$')
            ax.set_ylabel('$p(M)$')
            plt.savefig(self.output_events + '/event_{0}.pdf'.format(i+1), bbox_inches = 'tight')

    def compute_autocorrelation(self):
        """
        Computes autocorrelation function, defined as
        
            C(T) = (<x(t)x(t+T)> - <x>^2)/(<x(t)>^2-<x(t)^2>)
            
        where <•> denotes average over time.
        """
        sq_mean = np.mean(self.mass_samples)**2
        sq_std  = np.std(self.mass_samples)**2
        self.autocorrelation = []
        for i in range(int(len(self.mass_samples)/2.)):
            ac = 0.
            for j in range(len(self.mass_samples)):
                try:
                    ac += (self.mass_samples[j] * self.mass_samples[j+i]) - sq_mean
                except:
                    ac += (self.mass_samples[j] * self.mass_samples[(j+i) - len(self.mass_samples)]) - sq_mean
            self.autocorrelation.append(ac/(sq_std*len(self.mass_samples)))
            
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(self.autocorrelation, marker = '')
        ax.set_xlabel('$\\tau\ [a.u.]$')
        ax.set_ylabel('autocorrelation')
        fig.savefig(self.output_folder+'/autocorrelation.pdf', bbox_inches = 'tight')
 
    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """
        
        if self.verbose:
            self.display_config()
        self.run_sampling()
        # reconstructed events
        self.output_events = self.output_folder + '/reconstructed_events'
        if not os.path.exists(self.output_events):
            os.mkdir(self.output_events)
        self.plot_samples()
        self.output_samples_folder = self.output_folder + '/posterior_samples/'
        if not os.path.exists(self.output_samples_folder):
            os.mkdir(self.output_samples_folder)
        np.savetxt(self.output_samples_folder+'/mass_samples.txt', self.mass_samples)
        np.savetxt(self.output_samples_folder+'/sigma_samples.txt', self.sigma_samples)
        np.savetxt(self.output_samples_folder+'/alpha_samples.txt', self.alpha_samples)
        if self.diagnostic:
            self.compute_autocorrelation()
            
        # Samples
        plt.figure()
        ax1 = plt.subplot2grid((2,2), (0,0), colspan = 2)
        ax2 = plt.subplot2grid((2,2), (1,0), colspan = 1)
        ax3 = plt.subplot2grid((2,2), (1,1), colspan = 1)
        
        ax1.hist(self.mass_samples, bins = int(np.sqrt(len(self.mass_samples))), density = True)
        if self.injected_density is not None:
            app = np.linspace(self.min_m, self.max_m, 1000)
            ax1.plot(app, self.injected_density(app), c = 'red', label = 'Injected')
        ax1.set_xlabel('$M_1\ [M_\\odot]$')
        ax1.legend()
        ax2.hist(self.sigma_samples, bins = int(np.sqrt(len(self.sigma_samples))), density = True)
        ax2.set_xlabel('$\\sigma\ [M_\\odot]$')
        ax3.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))), density = True)
        ax3.set_xlabel('$\\alpha$')
        if not self.truths == None:
            truths_m_file, truths_s_file, truths_a_file = self.truths
            for m, s, a in zip(np.atleast_1d(np.genfromtxt(truths_m_file)), np.atleast_1d(np.genfromtxt(truths_s_file)), np.atleast_1d(np.genfromtxt(truths_a_file))):
                ax1.axvline(m, ls = '--', c = 'r', linewidth = 0.4)
                ax2.axvline(s, ls = '--', c = 'r', linewidth = 0.4)
                ax3.axvline(a, ls = '--', c = 'r', linewidth = 0.4)
        plt.tight_layout()
        plt.savefig(self.output_folder+'/posterior_samples.pdf', bbox_inches = 'tight')
        
        # acceptance
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(self.acceptance, marker = ',', ls = '')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acceptance')
        plt.savefig(self.output_folder+'/acceptance.pdf', bbox_inches = 'tight')
        
        # corner plot
        figure = corner.corner(np.column_stack((self.mass_samples, self.sigma_samples, self.alpha_samples)),
                           labels=[r"$M_1$", r"$\sigma$", r"$\alpha$"],
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True,
                           title_kwargs={"fontsize": 12})
        figure.savefig(self.output_folder+'/corner.pdf', bbox_inches = 'tight')
        return

    def postprocessing(self, samples_files = None):
        """
        Postprocesses set of pre-sampled data.
        
        -------------
        Arguments:
            :str samples_file:   Samples. If None, postprocesses data already stored in self.mass_samples.
        """
        
        if samples_files is not None:
            mass_file, sigma_file, alpha_file = samples_files
            self.mass_samples = np.genfromtxt(mass_file)
            self.sigma_samples = np.genfromtxt(sigma_file)
            self.alpha_samples = np.genfromtxt(alpha_file)
        
        # Samples
        plt.figure()
        ax1 = plt.subplot2grid((2,2), (0,0), colspan = 2)
        ax2 = plt.subplot2grid((2,2), (1,0), colspan = 1)
        ax3 = plt.subplot2grid((2,2), (1,1), colspan = 1)
        
        ax1.hist(self.mass_samples, bins = int(np.sqrt(len(self.mass_samples))), density = True, label = 'Posterior samples')
        if self.injected_density is not None:
            app = np.linspace(self.min_m, self.max_m, 1000)
            ax1.plot(app, self.injected_density(app), c = 'red', label = 'Injected')
        ax1.set_xlabel('$M_1\ [M_\\odot]$')
        ax1.legend()
        ax2.hist(self.sigma_samples, bins = int(np.sqrt(len(self.sigma_samples))), density = True)
        ax2.set_xlabel('$\\sigma\ [M_\\odot]$')
        ax3.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))), density = True)
        ax3.set_xlabel('$\\alpha$')
        if not self.truths == None:
            truths_m_file, truths_s_file, truths_a_file = self.truths
            for m, s, a in zip(np.genfromtxt(truths_m_file), np.genfromtxt(truths_s_file), np.genfromtxt(truths_a_file)):
                ax1.axvline(m, ls = '--', c = 'r', linewidth = 0.4)
                ax2.axvline(s, ls = '--', c = 'r', linewidth = 0.4)
                ax3.axvline(a, ls = '--', c = 'r', linewidth = 0.4)
        plt.savefig(self.output_folder+'/posterior_samples.pdf', bbox_inches = 'tight')

        # corner plot
        figure = corner.corner(np.column_stack((self.mass_samples, self.sigma_samples, self.alpha_samples)),
                           labels=[r"$M_1$", r"$\sigma$", r"$\alpha$"],
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True,
                           title_kwargs={"fontsize": 12})
        figure.savefig(self.output_folder+'/corner.pdf', bbox_inches = 'tight')
