import numpy as np
from numpy.random import uniform, randint
import numpy.random as rd
import matplotlib.pyplot as plt
import os
import corner
from numba import jit
from numba.experimental import jitclass
from numba import types, typed
from scipy.special import logsumexp
from scipy.stats import gamma, invgamma

class gibbs_dpgmm:

    def __init__(self,
                 events,
                 mass_b,
                 n_draws,
                 burnin,
                 step,
                 alpha0,
                 gamma0,
                 alpha = 3,
                 beta = 2,
                 V = 1,
                 max_stick = 16,
                 sigma_b = [np.log(2),np.log(20)],
                 output_folder = './',
                 delta_M = 1,
                 delta_s = 0.1,
                 injected_density = None,
                 verbose = True,
                 diagnostic = False):
        
        self.events  = events
        
        # Sampling options
        self.n_draws = n_draws
        self.burnin  = burnin
        self.step    = step
        
        # Boundaries (ma servono davvero?)
        self.max_m     = max(mass_b)
        self.min_m     = min(mass_b)
        self.max_sigma = max(sigma_b)
        self.min_sigma = min(sigma_b)

        
        # Normal-Inverse Gamma parameters
        # See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf eq (188)
        # Means are assumed to be E[] of each sample set
        self.alpha = alpha
        self.beta  = beta
        self.mu    = np.mean(events, axis = 1)
        self.V     = V
        
        # DPs
        self.max_stick = max_stick
        # Internal
        self.alpha0 = alpha0
        self.internal_base_distribution = [np.ones(self.max_stick+1)*len(samples)/self.max_stick for samples in self.events]
        for i in range(len(self.events)):
            self.internal_base_distribution[i][-1] = self.alpha0
        self.z_internal = [randint(0, self.max_stick, len(samples)) for samples in self.events]
        self.int_weights = [np.ones(self.max_stick)/self.max_stick for _ in self.events]
        self.int_means   = [np.ones(self.max_stick) for _ in self.events]
        self.int_sigmas  = [np.ones(self.max_stick) for _ in self.events]
        
        self.initialise_parameters()
        
        # External
        self.gamma0  = gamma0
        self.external_base_distribution = np.ones(self.max_stick+1)*len(events)/self.max_stick
        self.external_base_distribution[-1] = self.gamma0
        self.ext_weights = np.ones(self.max_stick)
        self.ext_means   = np.ones(self.max_stick)
        self.ext_sigmas  = np.ones(self.max_stick)
        
        # Posteriors
        self.internal_posterior_samples = [ [ [] for _ in range(self.max_stick)] for _ in range(len(self.events))]
        self.mass_samples = np.zeros(len(self.events))
        self.external_posterior_samples = [ [] for _ in range(self.max_stick)]
        
        # Miscellanea
        self.output_folder    = output_folder
        self.injected_density = injected_density
        self.verbose          = verbose
        self.diagnostic       = diagnostic
        
    def initialise_parameters(self):
        for i in range(len(self.events)):
            for j in range(self.max_stick):
                self.int_means[i][j], self.int_sigmas[i][j] = self.mean_sigma_sampler(i)

    def mean_sigma_sampler(self, event_index):
        sigma = self.beta*invgamma.rvs(self.alpha,1)
        mean  = rd.normal(self.mu[event_index], sigma*self.V)
        return [mean, sigma]
    
#    def sample_weights(self, event_index):
#        ys = np.zeros(self.max_stick+1)
#        for i in range(self.max_stick+1):
#            if self.internal_base_distribution[event_index][i] > 0:
#                ys[i] = gamma.rvs(self.internal_base_distribution[event_index][i])
#            else:
#                ys[i] = gamma.rvs(10**-2)
#        sum = np.sum(ys)
#        weights = np.zeros(self.max_stick)
#        for i in range(self.max_stick):
#            weights[i] = ys[i]/sum
#        return weights

    def sample_weights(self, event_index):
        betas = np.random.beta(1, self.alpha0, size=self.max_stick)
        betas[1:] *= np.cumprod(1 - betas[:-1])
        return betas
        
    def sample_z(self, s, event_index):
        probs = [np.exp(np.log(w) + self.log_normal_density(s, mu, sigma)) for w, mu, sigma in zip(self.int_weights[event_index], self.int_means[event_index], self.int_sigmas[event_index])]
        draw = uniform(0, np.sum(probs))
        tot = 0.
        for i in range(self.max_stick):
            tot += probs[i]
            if tot > draw:
                return i
    
    def log_normal_density(self, x, x0, sigma):
        """
        Normal probability density function.
        
        ------------
        Arguments:
            :double x:     Point.
            :double x0:    Mean.
            :double sigma: Variance.
        Returns:
            :double:       N(x).
        """
        return (-(x-x0)**2/(2*sigma**2))-np.log(np.sqrt(2*np.pi)*sigma)
    
    def evaluate_logP(self, samples, z, w, mu, sigma):
        logP = 0
        for si, zi in zip(samples, z):
            logP += np.log(w[zi]) + self.log_normal_density(si, mu[zi], sigma[zi])
        return logP
    
    def update_zs(self, event_index):
        for s, z, i in zip(self.events[event_index], self.z_internal[event_index], range(len(self.z_internal[event_index]))):
            new_z = self.sample_z(s, event_index)
            new_P = np.log(self.int_weights[event_index][new_z]) + self.log_normal_density(s, self.int_means[event_index][new_z], self.int_sigmas[event_index][new_z])
            old_P = np.log(self.int_weights[event_index][z]) + self.log_normal_density(s, self.int_means[event_index][z], self.int_sigmas[event_index][z])
            if new_P - old_P > np.log(uniform()):
                self.z_internal[event_index][i] = new_z
        
        for i in range(self.max_stick):
            self.internal_base_distribution[event_index][i] = np.sum(self.z_internal[event_index] == i)
        
    def update_proportions(self, event_index):
        new_w = self.sample_weights(event_index)
        old_P = self.evaluate_logP(self.events[event_index], self.z_internal[event_index], self.int_weights[event_index], self.int_means[event_index], self.int_sigmas[event_index])
        new_P = self.evaluate_logP(self.events[event_index], self.z_internal[event_index], new_w, self.int_means[event_index], self.int_sigmas[event_index])
        if new_P - old_P > np.log(uniform()):
            self.int_weights[event_index] = new_w
    
    def update_parameters(self, event_index):
        samples = self.events[event_index]
        for i in range(self.max_stick):
            new_P = 0.
            old_P = 0.
            new_m, new_s = self.mean_sigma_sampler(event_index)
            for index in np.where(self.z_internal == i)[0]:
                old_P += self.log_normal_density(samples[index], self.int_means[event_index][i], self.int_sigmas[event_index][i])
                new_P += self.log_normal_density(samples[index], new_m, new_s)
            if new_P - old_P > np.log(uniform()):
                self.int_means[event_index][i]  = new_m
                self.int_sigmas[event_index][i] = new_s
                
    def markov_step(self, event_index):
        self.update_zs(event_index)
        self.update_proportions(event_index)
        self.update_parameters(event_index)
        
    def save_posterior_samples(self, event_index):
        for j in range(self.max_stick):
            self.internal_posterior_samples[event_index][j].append({'mean': self.int_means[event_index][j], 'sigma': self.int_sigmas[event_index][j], 'weight': self.int_weights[event_index][j]})
                
    def run_sampling(self):
        for j in range(len(self.events)):
            for i in range(self.burnin):
                self.markov_step(j)
                if self.verbose:
                    print('\rBURN-IN: {0}/{1}'.format(i+1, self.burnin), end = '')
            if self.verbose:
                print('\n', end = '')
            print('burnin: done')
            for i in range(self.n_draws):
                if self.verbose:
                    print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws), end = '')
                for k in range(self.step):
                    self.markov_step(j)
                self.save_posterior_samples(j)
            if self.verbose:
                print('\n', end = '')
        
    def display_config(self):
        print('MCMC Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.events)))
        print('Mass interval: {0}-{1} Msun'.format(self.min_m, self.max_m))
        print('Concentration parameters:\nalpha0 = {0}\tgamma = {1}'.format(self.alpha0, self.gamma0))
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
        
        p = {}
        
        for samples, post, i in zip(self.events, self.internal_posterior_samples, range(len(self.events))):
            fig = plt.figure()
            fig.suptitle('Event {0}'.format(i+1))
            ax  = fig.add_subplot(111)
            ax.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True)
            probs = []
            for a in app:
                probs.append([logsumexp([self.log_normal_density(a, component['mean'], component['sigma']) + np.log(component['weight']) for component in sample]) for sample in post])
            for perc in percentiles:
                p[perc] = np.exp(np.percentile(probs, perc, axis = 1))
            np.savetxt(self.output_events + '/rec_prob_{0}.txt'.format(i+1), np.array(p[50]))
            
            ax.fill_between(app, p[95], p[5], color = 'lightgreen', alpha = 0.5)
            ax.fill_between(app, p[84], p[16], color = 'aqua', alpha = 0.5)
            ax.plot(app, p[50], marker = '', color = 'r')
            ax.set_xlabel('$M_1\ [M_\\odot]$')
            ax.set_ylabel('$p(M)$')
            plt.savefig(self.output_events + '/event_{0}.pdf'.format(i+1), bbox_inches = 'tight')
            
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
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        for i in range(len(self.events)):
            w = [np.mean([sample['weight'] for sample in component]) for component in self.internal_posterior_samples[i]]
            std = [np.std([sample['weight'] for sample in component]) for component in self.internal_posterior_samples[i]]
        ax.errorbar(np.arange(1, self.max_stick+1), w, yerr = std, ls = '', marker = '+')
        ax.set_xlabel('$w_i$')
        plt.savefig(self.output_folder+'/components.pdf', bbox_inches = 'tight')
#        for key in self.mass_samples.keys():
#            np.savetxt(self.output_samples_folder+'/mass_samples_{0}.txt'.format(key), self.mass_samples[key])
#        for key in self.sigma_samples.keys():
#            np.savetxt(self.output_samples_folder+'/sigma_samples_{0}.txt'.format(key), self.sigma_samples[key])
#        if self.diagnostic:
#            self.compute_autocorrelation()
        # samples
#        fig = plt.figure()
#        ax1  = fig.add_subplot(211)
#        ax2  = fig.add_subplot(212)
#        for key in self.mass_samples.keys():
#            fig.suptitle('Component #{0}'.format(key+1))
#            self.heights, self.bins, self.patches = ax1.hist(self.mass_samples[key], bins = int(np.sqrt(len(self.mass_samples[key]))), density = True, label = 'posterior samples')
#            if self.injected_density is not None:
#                app = np.linspace(self.min_m, self.max_m, 1000)
#                ax1.plot(app, self.injected_density(app), c = 'red', label = 'injected')
#                ax1.plot(app, [self.mass_prior(m) for m in app], c = 'green', label = 'prior')
#            ax1.set_xlabel('$M_1\ [M_\\odot]$')
#            ax2.hist(self.sigma_samples[key], bins = int(np.sqrt(len(self.sigma_samples[key]))), density = True, label = 'posterior samples')
#            ax2.set_xlabel('$\\sigma$')
#            plt.savefig(self.output_samples_folder+'/samples_{0}.pdf'.format(key+1), bbox_inches = 'tight')
#            ax1.clear()
#            ax2.clear()
        
#        # acceptance
#        fig = plt.figure()
#        ax1 = fig.add_subplot(211)
#        ax2 = fig.add_subplot(212)
#        fig.suptitle('Acceptance')
#        ax1.plot(self.acceptance_table, marker = ',', ls = '')
#        ax2.plot(self.acceptance_component, marker = ',', ls = '')
#        ax1.axvline(self.burnin, ls = '-.', lw = 0.3, color = 'r')
#        ax2.axvline(self.burnin, ls = '-.', lw = 0.3, color = 'r')
#        ax2.set_xlabel('Iteration')
#        ax1.set_ylabel('Table')
#        ax2.set_ylabel('Component')
#        plt.savefig(self.output_folder+'/acceptance.pdf', bbox_inches = 'tight')
        
        # corner plot
#        self.output_corner = self.output_folder+'/corner/'
#        if not os.path.exists(self.output_corner):
#            os.mkdir(self.output_corner)
#        for key in self.mass_samples.keys():
#            figure = corner.corner(np.column_stack((self.mass_samples[key], self.sigma_samples[key])),
#                           labels=[r"$M_1$", r"$\sigma$"],
#                           quantiles=[0.16, 0.5, 0.84],
#                           show_titles=True,
#                           title_kwargs={"fontsize": 12})
#            figure.savefig(self.output_corner+'/component_{0}.pdf'.format(key+1), bbox_inches = 'tight')
        return
