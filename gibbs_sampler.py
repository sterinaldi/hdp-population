import numpy as np
from numpy.random import uniform
import numpy.random as rd
import matplotlib.pyplot as plt
import os
import corner

class gibbs_sampler:

    """
    MCMC Gibbs sampler: Hierarchical Dirichlet Process.
    
    Based on Teh et al. (2006), sec. 5.1, follows the same notation.
    https://www.researchgate.net/publication/4742259_Hierarchical_Dirichlet_Processes
    ------------------
    Methods:
        __init__:                       Instances a new sampler.
        initalise_tables:               Draws a random mixture component for each sample x_i (n*m components).
        update_table:                   Markov step for a specific sample point.
        update_component:               Markov step for a specific mixture component.
        evaluate_probability_t:         Probability of drawing x_i given the fact that it belongs to a table t.
        evaluate_probability_component: Probability of drawing (x_i,...,x_k) given the fact that these samples belong to the table t.
        evaluate_probability_sample:    Probability of drawing x_i marginalized over all the possible components.
        normal_density:                 Normal probability density function.
        markov_step:                    Generates next MC point.
        save_mass_samples:              Saves generated mass samples in an appropriate list.
        run_sampling:                   Runs sampling process.
        single_bootstrap:               Single run of Bootstrap resampling algorithm for error estimation.
        bootstrap:                      Error estimation using Bootstrap technique.
        display_config:                 Prints configuration parameters.
        plot_samples:                   Plots x_i samples histogram along with inferred posterior distribution.
        compute_autocorrelation:        Computes autocorrelation function C(T).
        run:                            Runs the analysis. This and postprocessing() are potentially the only 'external' methods.
        postprocessing:                 Postprocesses pre-generated data.
        get_mass_samples:               Returns mass samples. Solves Ray usage issues.
    ------------------
    Usage example:
    
    import gibbs_sampler as GS
    from somewhere import normal
    
    events = []
    for event in event_folder:
        events.append(np.genfromtxt(events_path+event))
    output = '/some/output/path'
    mu     = 20.
    sigma  = 3.
    
    sampler = GS.gibbs_sampler.remote(samples = events,
                                      mass_b  = [5,50],
                                      n_draws = 1000,
                                      burnin  = 1000,
                                      step    = 10,
                                      alpha0  = 10,
                                      gamma   = 10,
                                      output_folder = output,
                                      verbose = True,
                                      injected_density = lambda x : normal(x, mu, sigma)
                                      )
    sampler.run()
    """
    
    def __init__(self,
                 samples,
                 mass_b,
                 n_draws,
                 burnin,
                 step,
                 alpha0,
                 gamma,
                 sigma_b = [np.log(2),np.log(4)],
                 output_folder = './',
                 n_resamples = 250,
                 delta_M = 1.,
                 injected_density = None,
                 verbose = True,
                 diagnostic = False):
        """
        Class instancer.
        
        -----------------
        Arguments:
            :np.array samples:        (n,m) shaped array containing n events, each of them with m samples x_i.
            :list mass_b:             Upper and lower prior mass boundaries.
            :int n_draws:             Number of generated samples.
            :int burnin:              Number of discarded samples (thermalization).
            :int step:                Number of markov steps between two subsequent samples (avoids autocorrelation).
            :int alpha0:              Concentration parameter for inner DP (DPGMM).
            :int gamma:               Concentration parameter for outer DP (mass function).
            :list sigma_b:            Upper and lower prior mass boundaries.
            :str output_folder        Output folder.
            :int n_resamples:         Number of bootstrap draws.
            :double delta_M:          Interval width around M_old for tentative sample drawing.
            :method injected_density: Injected probability density function, used for plotting purposes only. Optional.
            :bool verbose:            Printing statuts, default is True. Meant to be turned off while parallelizing.
            :bool diagnostic:         Diagnostic plots (autocorrelation)
            
        Return:
            :gibbs_sampler:           The instanciated sampler.
        """
        
        self.samples     = samples
        self.table_index  = []
        
        for i in range(len(samples)):
            self.table_index.append(list(np.zeros(len(samples[i]))))
            
        self.max_m     = max(mass_b)
        self.min_m     = min(mass_b)
        self.max_sigma = max(sigma_b)
        self.min_sigma = min(sigma_b)
        
        self.tables = []
        for i in range(len(samples)):
            self.tables.append([])
        self.components = []
        
        # Uniform prior on samples
        self.samples_prior = lambda x : 1/(self.max_m - self.min_m) if (self.min_m < x < self.max_m) else 0
        
        # Uniform prior on masses
        self.mass_prior = lambda x : 1/(self.max_m - self.min_m) if (self.min_m < x < self.max_m) else 0
        self.draw_mass  = lambda M: M + delta_M * uniform(-1,1)
        self.draw_mass_initial  = lambda : uniform(self.min_m, self.max_m)
        
        # Jeffreys prior on sigma
        self.sigma_prior = lambda x : 1/(x * np.log(self.max_sigma-self.min_sigma))
        self.draw_sigma  = lambda : np.exp(uniform(self.min_sigma,self.max_sigma))
        
        # Configuration parameters
        self.alpha0      = alpha0      # DPGMM concentration parameter
        self.gamma       = gamma       # Mass function concentration parameter
        self.n_draws     = n_draws     # total number of outcomes
        self.burnin      = burnin      # burn-in
        self.step        = step        # steps between two outcomes (avoids autocorrelation)
        self.n_resamples = n_resamples # bootstrap resamplings
        self.delta_M     = delta_M     # interval around old M sample (for updating)
        
        self.output_folder    = output_folder
        self.injected_density = injected_density
        self.verbose          = verbose
        self.diagnostic       = diagnostic
        
        self.posterior_samples = []
        self.mass_samples      = []
        self.sigma_samples     = []
        self.acceptance_table = []
        self.acceptance_component = []
        
        self.initialise_tables()
        return
        
        
    def initialise_tables(self):
        """
        Initialises a random mixture component for each sample x_i.
        """
        
        for j in range(len(self.table_index)):
            for i in range(len(self.table_index[j])):
                mass_temp  = self.draw_mass_initial()
                sigma_temp = self.draw_sigma()
                # Masses
                try:
                    index = self.components.index([mass_temp, sigma_temp])
                except:
                    self.components.append([mass_temp, sigma_temp])
                    index = self.components.index([mass_temp, sigma_temp])
                self.tables[j].append(index)
                self.table_index[j][i] = i
        return
    
    def update_table(self, sample_index, event_index):
        """
        Updates the mixture component the sample x_i is associated to.
        
        ---------------
        Arguments:
            :int sample_index: Sample index.
            :int event_index:  Event index.
        """
        
        flag_newtable     = False
        flag_newcomponent = False
    
        old_t         = int(self.table_index[event_index][sample_index])
        old_component = self.components[self.tables[event_index][old_t]]
        old_f         = self.normal_density(self.samples[event_index][sample_index], *old_component)
    
        # Generating new table
        
        # Selecting between choosing from existing tables or laying a new table
        if uniform() < self.alpha0/(self.alpha0 + len(self.tables[event_index])):
            new_t = int(max(self.table_index[event_index]) + 1)
            flag_newtable = True
            # If new table is instanciated, choose if sampling from existing parameters of generating a new set
            if uniform() < self.gamma/(self.gamma+len(self.components)):
                # new_component     = [self.draw_mass(), self.draw_sigma()]
                new_component     = [self.draw_mass(old_component[0]), self.draw_sigma()]
                flag_newcomponent = True
                new_f             = self.normal_density(self.samples[event_index][sample_index], *new_component)
                # Probability of generating sample x_i with new mixture component
                p_new = self.evaluate_probability_t(new_t, new_component, -1, sample_index, event_index, old_f, new_f)
            else:
                new_component = self.components[rd.choice(self.tables[rd.randint(low = 0, high = len(self.tables))])]
                new_f         = self.normal_density(self.samples[event_index][sample_index], *new_component)
                p_new = self.evaluate_probability_t(new_t, new_component, self.components.index(new_component), sample_index, event_index, old_f, new_f)
        else:
            new_t = int(rd.choice(self.table_index[event_index]))
            new_component = self.components[self.tables[event_index][new_t]]
            new_f = self.normal_density(self.samples[event_index][sample_index], *new_component)
            p_new = self.evaluate_probability_t(new_t, new_component, -1, sample_index, event_index, old_f, new_f)
        
        # Probabily of generating sample x_i with old mixture component
        p_old = self.evaluate_probability_t(old_t, old_component, self.components.index(old_component), sample_index, event_index, old_f, old_f)
        
        # Taking care of issue during burn-in (random draw of samples results in 0/0 probability ratio -> accept change)
        if p_old == 0.:
            p_new = 1.
            p_old = 0.5
            
        # Computing probability ratio and accepting with probability p = p_new/p_old
        if p_new/p_old > uniform():
            self.accept_table += 1.
            if flag_newtable:
                if flag_newcomponent:
                    self.components.append(new_component)
                self.tables[event_index].append(self.components.index(new_component))
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
        """
        Updates the mixture component the table t_i is associated to.
        
        ---------------
        Arguments:
            :int component_index: Component index.
            :int event_index:  Event index.
        """
        
        flag_newcomponent   = False
        old_component_index = self.tables[event_index][component_index]
        old_component       = self.components[old_component_index]
        
        # Selecting between choosing from an existing component or drawing a new component
        if uniform() < self.gamma/(self.gamma+len(self.components)):
            # new_component     = [self.draw_mass(), self.draw_sigma()]
            new_component     = [self.draw_mass(old_component[0]), self.draw_sigma()]
            flag_newcomponent = True
            # Probability of generating sample array [x] with new mixture component
            p_new = self.evaluate_probability_component(new_component, -1, event_index, self.samples[event_index])
        else:
            new_component = self.components[rd.choice(self.tables[rd.randint(low = 0, high = len(self.tables))])]
            p_new = self.evaluate_probability_component(new_component, self.components.index(new_component), event_index, self.samples[event_index])
        
        # Probability of generating sample array [x] with old mixture component
        p_old = self.evaluate_probability_component(old_component, old_component_index, event_index, self.samples[event_index])
        
        # taking care of issue during burn-in (random draw of samples results in 0/0 probability ratio -> accept change)
        if p_old == 0.:
            p_new = 1.
            p_old = 0.5
        
        # Computing probability ratio and accepting with probability p = p_new/p_old
        if p_new/p_old > uniform():
            self.accept_component += 1.
            if flag_newcomponent:
                self.components.append(new_component)
            self.tables[event_index][component_index] = self.components.index(new_component)
            if np.sum([table.count(old_component_index) for table in self.tables]) == 0:
                del self.components[old_component_index]
                self.tables = [[x-1 if x > old_component_index else x for x in table] for table in self.tables]
        return
    
    def evaluate_probability_t(self, table, component, component_index, sample_index, event_index, old_f, new_f):
        """
        Evaluates conditional distribution for a specific table.
        
        Eq. 32 in Teh et al. (2006).
        NB: since it requires old_f and new_f (see below), if computing old table probability just set new_f = old_f.
        --------------
        Arguments:
            :int table:           Table index.
            :list component:      Component parameters list.
            :int component_index: Component index.
            :int sample_index:    Sample index in event.
            :int event_index:     Event index.
            :double old_f:        Probability of generating sample x_i given the old mixture component.
            :double new_f:        Probability of generating sample x_i given the new mixture component.
        Returns:
            :double:              Computed probability.
        """
        
        n = self.table_index[event_index].count(table)
        if n == 0:
            return self.alpha0 * (self.evaluate_probability_sample(self.samples[event_index][sample_index], old_f, new_f))
        else:
            return (n-1) * self.normal_density(self.samples[event_index][sample_index], *component)
        
    def evaluate_probability_component(self, component, component_index, event_index, sample_array):
        """
        Evaluates conditional distribution for a specific component.
        
        Eq. 34 in Teh et al. (2006).
        --------------
        Arguments:
            :list component:      Component parameters list.
            :int component_index: Component index.
            :list sample_index:   Sample array.
            :int event_index:     Event index.
        Returns:
            :double:              Computed probability.
        """
        
        n = sum(table.count(component_index) for table in self.tables)
        if n == 0:
            return self.gamma * np.prod([self.samples_prior(x) for x in sample_array])
        else:
            return n * np.prod([self.normal_density(x, *component) for x in sample_array])
            
    def evaluate_probability_sample(self, sample, old_f, new_f):
        """
        Computes the probability of generating sample x_i given a set of mixture components.
        
        Eq. 31 in Teh et al. (2006).
        NB: since it requires old_f and new_f (see below), if computing old table probability just set new_f = old_f.
        ------------
        Arguments:
            :double sample: Sample x_i
            :double old_f:  Probability of generating sample x_i given the old mixture component.
            :double new_f:  Probability of generating sample x_i given the new mixture component.
        Returns:
            :double:        Computed probability.
        """
        
        return (np.sum([self.normal_density(sample, *self.components[index]) for table in self.tables for index in table]) + new_f - old_f + self.gamma * self.samples_prior(sample))/(np.sum([len(t) for t in self.tables])+self.gamma)
        
    def normal_density(self, x, x0, sigma):
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
        
        return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    
    
    def markov_step(self):
        """
        Computes the following Markov Chain point and computes acceptance.
        Here, acceptance - computed for tables and components apieces - is defined as n_changed_t/n_total_t or n_changed_k/n_total_k.
        """
        
        self.accept_table     = 0.
        self.accept_component = 0.
        tries_table    = 0.
        tries_component = 0.
        for event_index in range(len(self.samples)):
            for sample_index in range(len(self.samples[event_index])):
                self.update_table(sample_index, event_index)
                tries_table += 1.
            for component_index in range(len(self.tables[event_index])):
                self.update_component(component_index, event_index)
                tries_component += 1.
        self.acceptance_table.append(self.accept_table/tries_table)
        self.acceptance_component.append(self.accept_component/tries_component)
        return
    
    def save_posterior_samples(self):
        """
        Stores mass samples in apposite variable.
        """
        samples = []
        [self.components[index] for table in self.tables for index in table]
        for table_is, table in zip(self.table_index, self.tables):
            set_table_is = set(table_is)
            max = table[table_is[0]]
            for index in set_table_is:
                if table_is.count(index)/self.components[table[index]][1] > table_is.count(max)/self.components[table[max]][1]:
                    max = index
            samples.append(self.components[max])
        
        self.posterior_samples.append(samples)
    
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
        print('burnin: done')
        for i in range(self.n_draws):
            if self.verbose:
                print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for j in range(self.step):
                self.markov_step()
            self.save_posterior_samples()
        if self.verbose:
            print('\n', end = '')
        # flattening
        self.mass_samples = np.array([m[0] for draw in self.posterior_samples for m in draw])
        self.sigma_samples = np.array([m[1] for draw in self.posterior_samples for m in draw])
        self.posterior_samples = np.array([m for draw in self.posterior_samples for m in draw])
        return
    
    def single_bootstrap(self):
        """
        Single run of bootstrap algorithm.
        Produces an instance of heights.
        """
        
        indexes = rd.randint(low = 0, high = len(self.mass_samples), size = int(len(self.mass_samples)/4))
        samples = [self.mass_samples[i] for i in indexes]
        heights, bins, patches = plt.hist(samples, bins = self.bins, density = True)
        self.resampled_bins.append(heights)
        return
    
    def bootstrap(self):
        """
        Estimates bins variance using bootstrap technique.
        """
        
        self.resampled_bins = []
        for i in range(self.n_resamples):
            if self.verbose:
                print('\rBOOTSTRAP: {0}/{1}'.format(i+1, self.n_resamples), end = '')
            self.single_bootstrap()
        if self.verbose:
            print('\n', end = '')
        self.means  = np.array(self.resampled_bins).mean(axis = 0)
        self.errors = np.array(self.resampled_bins).std(axis = 0)
        return
        
    def display_config(self):
        print('MCMC Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.samples)))
        print('Mass interval: {0}-{1} Msun'.format(self.min_m, self.max_m))
        print('Concentration parameters:\nalpha0 = {0}\tgamma = {1}'.format(self.alpha0, self.gamma))
        print('Burn-in: {0} samples'.format(self.burnin))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws, self.step))
        print('Number of re-samples using Bootstrap technique: {0}'.format(self.n_resamples))
        print('------------------------')
        return
    
    def plot_samples(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution.
        """
        
        app = np.linspace(self.min_m, self.max_m, 1000)
        for samples, table_i, table, i in zip(self.samples, self.table_index, self.tables, range(len(self.samples))):
            fig = plt.figure()
            fig.suptitle('Event {0}'.format(i+1))
            ax  = fig.add_subplot(111)
            ax.hist(samples, bins = int(np.sqrt(len(samples))), density = True, color = 'lightblue')
            t = set(table_i)
            components = [self.components[table[t_i]] for t_i in t]
            ax.plot(app, [np.sum([self.normal_density(a, *component) * table_i.count(t_i)/len(table_i) for component, t_i in zip(components, t)]) for a in app], c = 'r')
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
        np.savetxt(self.output_folder+'/mass_samples.txt', self.mass_samples)
        if self.diagnostic:
            self.compute_autocorrelation()
        # samples
        fig = plt.figure(1)
        ax  = fig.add_subplot(111)
        self.heights, self.bins, self.patches = ax.hist(self.mass_samples, bins = int(np.sqrt(len(self.mass_samples))), density = True, label = 'posterior samples')
        if self.injected_density is not None:
            app = np.linspace(self.min_m, self.max_m, 1000)
            ax.plot(app, self.injected_density(app), c = 'red', label = 'injected')
            ax.plot(app, [self.mass_prior(m) for m in app], c = 'green', label = 'prior')
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        plt.savefig(self.output_folder+'/mass_samples.pdf', bbox_inches = 'tight')
        
        self.bootstrap()
        self.ref_bins = (self.bins + (self.bins[1]-self.bins[0])/2)[:-1]
        # bootstrapped samples
        fig = plt.figure(2)
        ax  = fig.add_subplot(111)
        ax.hist(self.mass_samples, bins = self.bins, alpha = 0.5, density = True)
        ax.fill_between(self.ref_bins, self.means+self.errors, self.means-self.errors, alpha=0.5, edgecolor='#3F7F4C', facecolor='aquamarine', label = 'posterior samples')
        if self.injected_density is not None:
            app = np.linspace(self.min_m, self.max_m, 1000)
            ax.plot(app, self.injected_density(app), c = 'red', label = 'injected')
            ax.plot(app, [self.mass_prior(m) for m in app], c = 'green', label = 'prior')
        ax.plot(self.ref_bins, self.heights, alpha = 0.5, ls = '--', c = 'yellow')
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        plt.savefig(self.output_folder+'/distribution.pdf', bbox_inches = 'tight')
        
        # reconstructed events
        self.output_events = self.output_folder + '/reconstructed_events'
        if not os.path.exists(self.output_events):
            os.mkdir(self.output_events)
        self.plot_samples()
        
        # acceptance
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        fig.suptitle('Acceptance')
        ax1.plot(self.acceptance_table, marker = ',', ls = '')
        ax2.plot(self.acceptance_component, marker = ',', ls = '')
        ax1.axvline(self.burnin, ls = '-.', lw = 0.3, color = 'r')
        ax2.axvline(self.burnin, ls = '-.', lw = 0.3, color = 'r')
        ax2.set_xlabel('Iteration')
        ax1.set_ylabel('Table')
        ax2.set_ylabel('Component')
        plt.savefig(self.output_folder+'/acceptance.pdf', bbox_inches = 'tight')
        
        # corner plot
        print(self.posterior_samples)
        fig = corner.corner(self.posterior_samples, labels=[r"$M_1$", r"$\sigma$"],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})
        fig.savefig(self.output_folder+'/corner_plot.pdf', bbox_inches = 'tight')
        return
    
    def postprocessing(self, samples_file = None, bootstrapping = False):
        """
        Postprocesses set of pre-sampled data.
        
        -------------
        Arguments:
            :str samples_file:   Samples. If None, postprocesses data already stored in self.mass_samples.
            :bool bootstrapping: If True, runs bootstrap
        """
        
        if samples_file is not None:
            self.mass_samples = np.genfromtxt(samples_file)
        # samples
        fig = plt.figure(1)
        ax  = fig.add_subplot(111)
        self.heights, self.bins, self.patches = ax.hist(self.mass_samples, bins = int(np.sqrt(len(self.mass_samples))), density = True, label = 'posterior samples')
        if self.injected_density is not None:
            app = np.linspace(self.min_m, self.max_m, 1000)
            ax.plot(app, self.injected_density(app), c = 'red', label = 'injected')
            ax.plot(app, [self.mass_prior(m) for m in app], c = 'green', label = 'prior')
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        plt.savefig(self.output_folder+'/mass_samples.pdf', bbox_inches = 'tight')
        if bootstrapping:
            self.bootstrap()
        
        self.ref_bins = (self.bins + (self.bins[1]-self.bins[0])/2)[:-1]
        # bootstrapped samples
        fig = plt.figure(2)
        ax  = fig.add_subplot(111)
        ax.hist(self.mass_samples, bins = self.bins, alpha = 0.5, density = True)
        ax.fill_between(self.ref_bins, self.means+self.errors, self.means-self.errors, alpha=0.5, edgecolor='#3F7F4C', facecolor='aquamarine', label = 'posterior samples')
        if self.injected_density is not None:
            app = np.linspace(self.min_m, self.max_m, 1000)
            ax.plot(app, self.injected_density(app), c = 'red', label = 'injected')
            ax.plot(app, [self.mass_prior(m) for m in app], c = 'green', label = 'prior')
        ax.plot(self.ref_bins, self.heights, alpha = 0.5, ls = '--', c = 'yellow')
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        plt.savefig(self.output_folder+'/distribution.pdf', bbox_inches = 'tight')
        return

    def get_mass_samples(self):
        """
        Returns samples. Solves issue with Ray parallelization.
        
        -------------
        Returns:
            :list: Mass samples.
        """
        
        return self.mass_samples
