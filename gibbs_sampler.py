import numpy as np
from numpy.random import uniform
import random as rd
import matplotlib.pyplot as plt
import os

class gibbs_sampler:
    
    def __init__(self,
                 samples,
                 mass_boundaries,
                 sigma_boundaries,
                 n_draws,
                 burnin,
                 step,
                 alpha0,
                 gamma,
                 output_folder = os.getcwd(),
                 n_resamples = 250):
        
        self.samples     = samples
        self.table_index  = []
        
        for i in range(len(samples)):
            self.table_index.append(np.zeros(len(samples[i])))
            
        self.max_m     = max(mass_boundaries)
        self.min_m     = min(mass_boundaries)
        self.max_sigma = max(sigma_boundaries)
        self.min_sigma = min(sigma_boundaries)
        
        tables = []
        for i in range(len(samples)):
            tables.append([])
        self.components = []
        
        # Uniform prior on samples
        self.samples_prior = lambda x : 1/(max(masses)-min(masses)) if (min(masses) < x < max(masses)) else 0
        
        # Uniform prior on masses
        self.mass_prior = lambda x : 1/(max(masses)-min(masses)) if (min(masses) < x < max(masses)) else 0
        self.draw_mass  = lambda : uniform(self.min_m, self.max_m)
        
        # Jeffreys prior on sigma
        self.sigma_prior = lambda x : 1/(x * np.log(self.max_sigma-self.min_sigma))
        self.draw_sigma  = lambda : np.exp(uniform(self.min_sigma,self.max_sigma))
        
        # Configuration parameters
        self.alpha0      = alpha0
        self.gamma       = gamma
        self.n_draws     = n_draws     # total number of outcomes
        self.burnin      = burnin      # burn-in
        self.step        = step        # steps between two outcomes (avoids autocorrelation)
        self.n_resamples = n_resamples # bootstrap resamplings
        
        self.mass_samples = []
        self.initialise_tables()
        
        return
        
        
    def initialise_tables(self):
        for j in len(self.table_index):
            for i in range(len(self.table_index[j])):
                mass_temp  = self.draw_mass()
                sigma_temp = self.draw_sigma()
                # Masses
                try:
                    index = self.components.index([mass_temp, sigma_temp])
                except:
                    self.components.append([mass_temp, sigma_temp])
                    index = self.components.index([mass_temp, sigma_temp])
                self.tables[i].append(index)
                self.table_index[j][i] = i
        return
    
    def update_table(self, sample_index, event_index):
        
        flag_newtable     = False
        flag_newcomponent = False
        
        # Choosing between new t and old t
        old_t         = self.table_index[event_index][sample_index]
        old_component = self.components[self.tables[event_index][old_t]]
 
        if uniform() < self.alpha0/(self.alpha0 + len(self.samples[event_index])):
            new_t = max(self.table_index[event_index]) + 1
            flag_newtable = True
            if uniform() < self.gamma/(self.gamma+len(self.components)):
                new_component     = [self.draw_mass(), self.draw_sigma()]
                flag_newcomponent = True
                p_new = self.evaluate_probability_t(new_t, new_component, -1, sample_index, event_index)
            else:
                new_component = self.components[rd.choice(rd.choice(self.tables))]
                p_new = self.evaluate_probability_t(new_t, new_component, self.components.index(new_component) sample_index, event_index)
        else:
            new_t = choice(self.table_index[event_index])
            new_component = self.components[self.tables[event_index][new_t]]
            p_new = self.evaluate_probability_t(new_t, new_component, sample_index, event_index)
        
        p_old = self.evaluate_probability_t(old_t, old_component, self.component.index(old_component), sample_index, event_index)
        
        if p_new/p_old > uniform():
            if flag_newtable:
                if flag_newcomponent:
                    self.components.append(new_component)
                self.tables[event_index].append(self.component.index(new_component))
            self.table_index[event_index][sample_index] = new_t
            
            if self.table_index[event_index].count(old_t) == 0:
                old_component_index = self.tables[event_index][old_t]
                del self.tables[event_index][old_t]
                if np.sum([table.count(old_component_index) for table in self.tables]) == 0:
                    del self.components[old_component_index]
                    self.tables = [[x-1 if x > old_component_index else x for x in table] for table in self.tables]
                self.table_index[event_index] = [x-1 if x > old_t else x for x in self.table_index[event_index]]
                    
        
        return
        
    def update_component(self, component_index, event_index):
        
        flag_newcomponent   = False
        old_component_index = self.tables[event_index][component_index]
        old_component       = self.components[old_component_index]
        
        if uniform() < self.gamma/(self.gamma+len(self.components)):
            new_component     = [self.draw_mass, self.draw_sigma]
            flag_newcomponent = True
            p_new = self.evaluate_probability_component(new_component, -1, event_index, self.samples[event_index])
        else:
            new_component = self.components[rd.choice(rd.choice(self.tables))]
            p_new = self.evaluate_probability_component(new_component, self.components.index(new_component), event_index, self.samples[event_index])
        
        p_old = self.evaluate_probability_component(old_component, old_component_index, event_index, self.samples[event_index])
        
        if p_new/p_old > uniform():
            if flag_newcomponent:
                self.components.append(new_component)
            self.tables[event_index][component_index] = self.components.index(new_component)
            if np.sum([table.count(old_component_index) for table in self.tables]) == 0:
                del self.components[old_component_index]
                self.tables = [[x-1 if x > old_component_index else x for x in table] for table in self.tables]å
        return

    def evaluate_probability_t(self, table, component, component_index, sample_index, event_index):
        n = self.table_index[event_index].count(table)
        if n == 0:
            return self.alpha0 * self.evaluate_probability_sample(self.samples[event_index][sample_index]) * self.evaluate_probability_component(component, component_index, event_index, [self.samples[event_index][sample_index]])
        else:
            return n * normal_density(self.samples[event_index][sample_index], *component)
        
    def evaluate_probability_component(self, component, component_index, event_index, sample_array):
        n = sum(table.count(component_index) for table in tables)
        if n == 0:
            return self.gamma * np.prod([normal_density(x, *component) for x in sample_array])
        else:
            return n * np.prod([normal_density(x, *component) for x in sample_vector])
            
    def evaluate_probability_sample(self, sample):
        return (np.sum([normal_density(sample, *self.components(index)) for table in self.tables for index in table]) + self.gamma * self.samples_prior(sample))/(np.sum([len(t) for t in self.tables])+self.gamma)
        
    def normal_density(x, x0, sigma):
        return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    
    
    def markov_step(self):
        for event_index in range(len(samples)):
            for sample_index in range(len(samples[event_index])):
                self.update_table(sample_index, event_index)
            for component_index in range(len(tables[event_index])):
                self.update_component(component_index, event_index)
        return
    
    def save_mass_samples(self):
        self.mass_samples.append([self.components[index][0] for table in self.tables for index in table])
    
    def run_sampling(self):
        for i in range(self.burnin):
            self.markov_step()
            print('\rBURN-IN: {0}/{1}'.format(i+1, self.burnin))
        print('\n')
        for i in range(self.n_draws):
            print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws))
            for j in range(self.step):
                self.markov_step()
            self.save_mass_samples()
            print('\n')
        self.mass_samples = np.array([m for draw in self.mass_samples for m in draw])
        return
    
    def single_bootstrap(self):
        samples = choice(self.mass_samples, len(self.mass_samples))
        heights, bins, patches = plt.hist(samples, bins = self.bins, density = True)
        self.resampled_bins.append(heights)
        return
    
    def bootstrap(self):
        self.resampled_bins = []
        for i in range(self.n_resamples):
            print('\rBOOTSTRAP: {0}/{1}'.format(i, self.n_resamples))
            self.single_bootstrap()
        print('\n')
        self.means  = np.array(self.resampled_bins).mean(axis = 1)
        self.errors = np.array(self.resampled_bins).std(axis = 1)
        return
    
    def run(self):
        self.run_sampling()
        np.savetxt(self.output+'/mass_samples.txt', self.mass_samples)
        
        # samples
        fig = plt.figure(1)
        ax  = fig.add_subplot(111)
        self.heights, self.bins, self.patches = ax.hist(self.mass_samples, bins = 'auto', density = True)
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        plt.savefig(self.output+'/mass_samples.pdf', bbox_inches = 'tight')
        
        self.ref_bins = (self.bins + (self.bins[1]-self.bins[0])/2)[:-1]
        self.bootstrap()
        # bootstrapped samples
        fig = plt.figure(2)
        ax  = ax.fig.add_subplot(111)
        ax.fill_between(self.ref_bins, self.means+self.errors, self.means-self.errors, alpha=0.5, edgecolor='#3F7F4C', facecolor='aquamarine')
        ax.plot(self.ref_bins, self.heights, ls = '--', c = 'green')
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        plt.savefig(self.output+'/distribution.pdf', bbox_inches = 'tight')
    
    def postprocessing(self, samples_file = None, bootstrapping = False):
        if samples_file is not None:
            self.mass_samples = np.genfromtxt(samples_file)
        if bootstrapping:
            self.bootstrap()
        # samples
        fig = plt.figure(1)
        ax  = fig.add_subplot(111)
        self.heights, self.bins, self.patches = ax.hist(self.mass_samples, bins = 'auto', density = True)
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        plt.savefig(self.output+'/mass_samples.pdf', bbox_inches = 'tight')
        
        self.ref_bins = (self.bins + (self.bins[1]-self.bins[0])/2)[:-1]
        self.bootstrap()
        # bootstrapped samples
        fig = plt.figure(2)
        ax  = ax.fig.add_subplot(111)
        ax.fill_between(self.ref_bins, self.means+self.errors, self.means-self.errors, alpha=0.5, edgecolor='#3F7F4C', facecolor='aquamarine')
        ax.plot(self.ref_bins, self.heights, ls = '--', c = 'green')
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        plt.savefig(self.output+'/distribution.pdf', bbox_inches = 'tight')
