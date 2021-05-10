import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings

from collections import namedtuple, Counter
from numpy import random

from scipy import stats
from scipy.stats import t as student_t
from scipy.stats import entropy, gamma
from scipy.special import logsumexp, betaln, gammaln
from scipy.interpolate import interp1d
from scipy.integrate import dblquad

from sampler_component_pars import sample_point

import mpmath as mp

from time import perf_counter
from itertools import product

from numba import jit, njit
from numba import prange
import ray
from ray.util import ActorPool
from multiprocessing import Pool
#from ray.util.multiprocessing import Pool

import pickle


"""
Implemented as in https://dp.tdhopper.com/collapsed-gibbs/
Modified (marginalisation)
Interesting answer: https://stats.stackexchange.com/questions/330488/estimate-the-variance-of-gaussian-distribution-from-noisy-sample
"""

# natural sorting.
# list.sort(key = natural_keys)

def sort_matrix(a, axis = -1):
    mat = np.array([[m, f] for m, f in zip(a[0], a[1])])
    keys = np.array([x for x in mat[:,axis]])
    sorted_keys = np.copy(keys)
    sorted_keys = np.sort(sorted_keys)
    indexes = [np.where(el == keys)[0][0] for el in sorted_keys]
    sorted_mat = np.array([mat[i] for i in indexes])
    return sorted_mat[:,0], sorted_mat[:,1]
    


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class CGSampler:
                
    def __init__(self, events,
                       samp_settings, # burnin, draws, step (list)
                       samp_settings_ev = None,
                       mass_chain_settings = [100,10], # burnin masses, step masses
                       alpha0 = 1,
                       gamma0 = 1,
                       hyperpars_ev = [1,3,1/4.],
                       hyperpars = [1,10,1/4.], # a, b, V
                       m_min = 5,
                       m_max = 70,
                       verbose = True,
                       output_folder = './',
                       initial_cluster_number = 5.,
                       process_events = True,
                       n_parallel_threads = 8,
                       injected_density = None,
                       true_masses = None,
                       diagnostic = False,
                       sigma_max  = 4,
                       sigma_max_ev = None,
                       names = None,
                       autocorrelation = False,
                       autocorrelation_ev = False
                       ):
        
        self.events = events
        self.burnin_mf, self.n_draws_mf, self.step_mf = samp_settings
        if samp_settings_ev is not None:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings_ev
        else:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings
        self.burnin_masses, self.step_masses = mass_chain_settings

        sample_min = np.min([np.min(a) for a in self.events])
        sample_max = np.max([np.max(a) for a in self.events])
        self.m_min   = min([m_min, sample_min])
        if self.m_min < 0.01:
            self.m_min = 0.01
        self.m_max   = max([m_max, sample_max])
        self.m_max_plot = m_max
        # DP
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        # student-t
        self.a_mf, self.b_mf, self.V_mf = hyperpars
        if hyperpars_ev is not None:
            self.a_ev, self.b_ev, self.V_ev = hyperpars_ev
        else:
            self.a_ev, self.b_ev, self.V_ev = hyperpars
        self.sigma_max = sigma_max
        if sigma_max_ev is not None:
            self.sigma_max_ev = sigma_max_ev
        else:
            self.sigma_max_ev = sigma_max
        # miscellanea
        self.output_folder = output_folder
        self.icn = initial_cluster_number
        self.event_samplers = []
        self.delta_M = [np.std(e) for e in self.events]
        self.verbose = verbose
        self.process_events = process_events
        self.diagnostic = diagnostic
        self.n_parallel_threads = n_parallel_threads
        self.injected_density = injected_density
        self.true_masses = true_masses
        self.output_recprob = self.output_folder + '/reconstructed_events/pickle/'
        if names is not None:
            self.names = names
        else:
            self.names = [str(i+1) for i in range(len(self.events))]
        self.autocorrelation = autocorrelation
        self.autocorrelation_ev = autocorrelation_ev
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        
    def initialise_samplers(self, marker):
        event_samplers = []
        for i, ev in enumerate(self.events[marker:marker+self.n_parallel_threads]):
            event_samplers.append(Sampler_SE.remote(
                                            ev,
                                            self.names[marker+i],
                                            self.burnin_ev,
                                            self.n_draws_ev,
                                            self.step_ev,
                                            self.alpha0,
                                            self.b_ev,
                                            self.a_ev,
                                            self.V_ev,
                                            self.m_min,
                                            self.m_max,
                                            self.output_folder,
                                            False,
                                            self.icn,
                                            self.sigma_max_ev,
                                            self.autocorrelation_ev
                                            ))
        return event_samplers
        
    def run_event_sampling(self):
        i = 0
        self.posterior_functions_events = []
        for n in range(int(len(self.events)/self.n_parallel_threads)+1):
            tasks = self.initialise_samplers(n*self.n_parallel_threads)
            pool = ActorPool(tasks)
            for s in pool.map(lambda a, v: a.run.remote(), range(len(tasks))):
                self.posterior_functions_events.append(s)
                i += 1
                print('\rProcessed {0}/{1} events\r'.format(i, len(self.events)), end = '')
        return
    
    def load_mixtures(self):
        print('Loading mixtures...')
        self.posterior_functions_events = []
        prob_files = [self.output_recprob+f for f in os.listdir(self.output_recprob) if f.startswith('posterior_functions')]
        prob_files.sort(key = natural_keys)
        for prob in prob_files:
            sampfile = open(prob, 'rb')
            samps = pickle.load(sampfile)
            self.posterior_functions_events.append(samps)
    
    def display_config(self):
        print('Collapsed Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.events)))
        print('Concentration parameters:\nalpha0 = {0}\tgamma0 = {1}'.format(self.alpha0, self.gamma0))
        print('Burn-in: {0} samples'.format(self.burnin_mf))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws_mf, self.step_mf))
        print('------------------------')
        return
    
    def run_mass_function_sampling(self):
        ray.shutdown()
        self.load_mixtures()
        self.mf_folder = self.output_folder+'/mass_function/'
        if not os.path.exists(self.mf_folder):
            os.mkdir(self.mf_folder)
            
        sampler = MF_Sampler(self.posterior_functions_events,
                       self.burnin_mf,
                       self.n_draws_mf,
                       self.step_mf,
                       delta_M = self.delta_M,
                       alpha0 = self.gamma0,
                       m_min = self.m_min,
                       m_max = self.m_max,
                       verbose = self.verbose,
                       output_folder = self.mf_folder,
                       initial_cluster_number = min([self.icn, len(self.posterior_functions_events)]),
                       injected_density = self.injected_density,
                       true_masses = self.true_masses,
                       diagnostic = self.diagnostic,
                       sigma_max = self.sigma_max,
                       m_max_plot = self.m_max_plot,
                       autocorrelation = self.autocorrelation,
                       n_parallel_threads = self.n_parallel_threads
                       )
        
        sampler.run()
    
    def run(self):
        init_time = perf_counter()
        self.display_config()
        if self.process_events:
            self.run_event_sampling()
        self.run_mass_function_sampling()
        end_time = perf_counter()
        seconds = int(end_time - init_time)
        h = int(seconds/3600.)
        m = int((seconds%3600)/60)
        s = int(seconds - h*3600-m*60)
        print('Elapsed time: {0}h {1}m {2}s'.format(h, m, s))
        return
        

def my_student_t(df, t):
    b = betaln(0.5, df*0.5)
    return -0.5*np.log(df*np.pi)-b-((df+1)*0.5)*np.log1p(t*t/df)
    
@ray.remote
class Sampler_SE:
    def __init__(self, mass_samples,
                       event_id,
                       burnin,
                       n_draws,
                       step,
                       alpha0 = 1,
                       b = 5,
                       a = 3,
                       V = 1/4.,
                       m_min = 5,
                       m_max = 50,
                       output_folder = './',
                       verbose = True,
                       initial_cluster_number = 5.,
                       sigma_max = 5.,
                       autocorrelation = False
                       ):
        # New seed for each subprocess
        random.RandomState(seed = os.getpid())
        self.mass_samples  = mass_samples
        self.e_ID    = event_id
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.m_min   = m_min
        self.m_max   = m_max
        if sigma_max < (max(mass_samples) - min(mass_samples))/3.:
            self.sigma_max = sigma_max
        else:
            self.sigma_max = (max(mass_samples) - min(mass_samples))/3.
        # DP parameters
        self.alpha0 = alpha0
        # Student-t parameters
        self.b  = a*(b**2)
        self.a  = a
        self.V  = V
        self.mu = np.mean(mass_samples)
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        self.SuffStat = namedtuple('SuffStat', 'mean var N')
        # Output
        self.output_folder = output_folder
        self.mixture_samples = []
        self.n_clusters = []
        self.verbose = verbose
        self.autocorrelation = autocorrelation
        self.alpha_samples = []
        
    def initial_state(self, samples):
        assign = [a%int(self.icn) for a in range(len(samples))]
        cluster_ids = list(np.arange(int(np.max(assign)+1)))
        samp = np.copy(samples)
        state = {
            'cluster_ids_': cluster_ids,
            'data_': samp,
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'Ntot': len(samples),
            'hyperparameters_': {
                "b": self.b,
                "a": self.a,
                "V": self.V,
                "mu": self.mu
                },
            'suffstats': {cid: None for cid in cluster_ids},
            'assignment': assign,
            'pi': {cid: self.alpha0 / self.icn for cid in cluster_ids},
            }
        self.update_suffstats(state)
        return state
    
    def update_suffstats(self, state):
        for cluster_id, N in Counter(state['assignment']).items():
            points_in_cluster = [x for x, cid in zip(state['data_'], state['assignment']) if cid == cluster_id]
            mean = np.array(points_in_cluster).mean()
            var  = np.array(points_in_cluster).var()
            M    = len(points_in_cluster)
            state['suffstats'][cluster_id] = self.SuffStat(mean, var, M)
    
    def log_predictive_likelihood(self, data_id, cluster_id, state):
        
        if cluster_id == "new":
            ss = self.SuffStat(0,0,0)
        else:
            ss  = state['suffstats'][cluster_id]
            
        x = state['data_'][data_id]
        mean = ss.mean
        sigma = ss.var
        N     = ss.N
        # Update hyperparameters
        V_n  = 1/(1/state['hyperparameters_']["V"] + N)
        mu_n = (state['hyperparameters_']["mu"]/state['hyperparameters_']["V"] + N*mean)*V_n
        b_n  = state['hyperparameters_']["b"] + (state['hyperparameters_']["mu"]**2/state['hyperparameters_']["V"] + (sigma + mean**2)*N - mu_n**2/V_n)/2.
        a_n  = state['hyperparameters_']["a"] + N/2.
        # Update t-parameters
        t_sigma = np.sqrt(b_n*(1+V_n)/a_n)
        t_sigma = min([t_sigma, self.sigma_max])
        t_x     = (x - mu_n)/t_sigma
        # Compute logLikelihood
        logL = my_student_t(df = 2*a_n, t = t_x)
        return logL

    def add_datapoint_to_suffstats(self, x, ss):
        mean = (ss.mean*(ss.N)+x)/(ss.N+1)
        var  = (ss.N*(ss.var + ss.mean**2) + x**2)/(ss.N+1) - mean**2
        return self.SuffStat(mean, var, ss.N+1)


    def remove_datapoint_from_suffstats(self, x, ss):
        if ss.N == 1:
            return(self.SuffStat(0,0,0))
        mean = (ss.mean*(ss.N)-x)/(ss.N-1)
        var  = (ss.N*(ss.var + ss.mean**2) - x**2)/(ss.N-1) - mean**2
        return self.SuffStat(mean, var, ss.N-1)
    
    def cluster_assignment_distribution(self, data_id, state):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster.
        """
        scores = {}
        cluster_ids = list(state['suffstats'].keys()) + ['new']
        for cid in cluster_ids:
            scores[cid] = self.log_predictive_likelihood(data_id, cid, state)
            scores[cid] += self.log_cluster_assign_score(cid, state)
        scores = {cid: np.exp(score) for cid, score in scores.items()}
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores

    def log_cluster_assign_score(self, cluster_id, state):
        """Log-likelihood that a new point generated will
        be assigned to cluster_id given the current state.
        """
        if cluster_id == "new":
            return np.log(state["alpha_"])
        else:
            return np.log(state['suffstats'][cluster_id].N)

    def create_cluster(self, state):
        state["num_clusters_"] += 1
        cluster_id = max(state['suffstats'].keys()) + 1
        state['suffstats'][cluster_id] = self.SuffStat(0, 0, 0)
        state['cluster_ids_'].append(cluster_id)
        return cluster_id

    def destroy_cluster(self, state, cluster_id):
        state["num_clusters_"] -= 1
        del state['suffstats'][cluster_id]
        state['cluster_ids_'].remove(cluster_id)
        
    def prune_clusters(self,state):
        for cid in state['cluster_ids_']:
            if state['suffstats'][cid].N == 0:
                self.destroy_cluster(state, cid)

    def sample_assignment(self, data_id, state):
        """
        Sample new assignment from marginal distribution.
        If cluster is "`new`", create a new cluster.
        """
        scores = self.cluster_assignment_distribution(data_id, state).items()
        labels, scores = zip(*scores)
        cid = random.RandomState().choice(labels, p=scores)
        if cid == "new":
            return self.create_cluster(state)
        else:
            return int(cid)

    def update_alpha(self, state, trimming = 100):
        a_old = state['alpha_']
        n     = state['Ntot']
        K     = len(state['cluster_ids_'])
        for _ in range(trimming):
            a_new = random.RandomState().gamma(1)
            logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old)
            logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new)
            if logP_new - logP_old > np.log(random.uniform()):
                a_old = a_new
        return a_old

    def gibbs_step(self, state):
        """
        Collapsed Gibbs sampler for Dirichlet Process Mixture Model
        """
        # alpha sampling
        state['alpha_'] = self.update_alpha(state)
        self.alpha_samples.append(state['alpha_'])
        pairs = zip(state['data_'], state['assignment'])
        for data_id, (datapoint, cid) in enumerate(pairs):
            state['suffstats'][cid] = self.remove_datapoint_from_suffstats(datapoint, state['suffstats'][cid])
            self.prune_clusters(state)
            cid = self.sample_assignment(data_id, state)
            state['assignment'][data_id] = cid
            state['suffstats'][cid] = self.add_datapoint_to_suffstats(state['data_'][data_id], state['suffstats'][cid])
        self.n_clusters.append(len(state['cluster_ids_']))
    
    def sample_mixture_parameters(self, state):
        ss = state['suffstats']
        alpha = [ss[cid].N + state['alpha_'] / state['num_clusters_'] for cid in state['cluster_ids_']]
        weights = random.RandomState().dirichlet(alpha).flatten()
        components = {}
        for i, cid in enumerate(state['cluster_ids_']):
            mean = ss[cid].mean
            sigma = ss[cid].var
            N     = ss[cid].N
            V_n  = 1/(1/state['hyperparameters_']["V"] + N)
            mu_n = (state['hyperparameters_']["mu"]/state['hyperparameters_']["V"] + N*mean)*V_n
            b_n  = state['hyperparameters_']["b"] + (state['hyperparameters_']["mu"]**2/state['hyperparameters_']["V"] + (sigma + mean**2)*N - mu_n**2/V_n)/2.
            a_n  = state['hyperparameters_']["a"] + N/2.
            # Update t-parameters
            s = stats.invgamma(a_n, scale = b_n).rvs()
            m = stats.norm(mu_n, s*V_n).rvs()
            components[i] = {'mean': m, 'sigma': np.sqrt(s), 'weight': weights[i]}
        self.mixture_samples.append(components)
    
    def run_sampling(self):
        state = self.initial_state(self.mass_samples)
        for i in range(self.burnin):
            if self.verbose:
                print('\rBURN-IN: {0}/{1}'.format(i+1, self.burnin), end = '')
            self.gibbs_step(state)
        if self.verbose:
            print('\n', end = '')
        for i in range(self.n_draws):
            if self.verbose:
                print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for _ in range(self.step):
                self.gibbs_step(state)
            self.sample_mixture_parameters(state)
        if self.verbose:
            print('\n', end = '')
        return
    
    def display_config(self):
        print('MCMC Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.events)))
        print('Concentration parameters:\nalpha0 = {0}\tgamma0 = {1}'.format(self.alpha0, self.gamma0))
        print('Burn-in: {0} samples'.format(self.burnin))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws, self.step))
        print('------------------------')
        return

    def postprocess(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution.
        """
        
        app  = np.linspace(self.m_min, self.m_max, 1000)
        percentiles = [5,16, 50, 84, 95]
        
        p = {}
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.hist(self.mass_samples, bins = int(np.sqrt(len(self.mass_samples))), histtype = 'step', density = True)
        prob = []
        for a in app:
            prob.append([logsumexp([log_normal_density(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) for sample in self.mixture_samples])
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        
        prob = np.array(prob)
        
        ent = []
        
        for i in range(np.shape(prob)[1]):
            sample = np.exp(prob[:,i])
            ent.append(entropy(sample,p[50]))
        mean_ent = np.mean(ent)
        np.savetxt(self.output_entropy + '/KLdiv_{0}.txt'.format(self.e_ID), np.array(ent), header = 'mean entropy = {0}'.format(mean_ent))
        
        picklefile = open(self.output_pickle + '/posterior_functions_{0}.pkl'.format(self.e_ID), 'wb')
        pickle.dump(self.mixture_samples, picklefile)
        picklefile.close()
        
        self.sample_probs = prob
        self.median_mf = np.array(p[50])
        
        ax.fill_between(app, p[95], p[5], color = 'lightgreen', alpha = 0.5)
        ax.fill_between(app, p[84], p[16], color = 'aqua', alpha = 0.5)
        ax.plot(app, p[50], marker = '', color = 'r')
        np.savetxt(self.output_recprob + '/log_rec_prob_{0}.txt'.format(self.e_ID), np.array([app, np.log(p[50])]).T)
        ax.set_xlabel('$M\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        ax.set_xlim(min(self.mass_samples)-5, max(self.mass_samples)+5)
        plt.savefig(self.output_pltevents + '/{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        fig = plt.figure()
        for i, s in enumerate(self.mixture_samples[:25]):
            ax = fig.add_subplot(5,int(len(self.mixture_samples[:25])/5),i+1)
            app = np.linspace(min(self.mass_samples),max(self.mass_samples),1000)
            for c in s.values():
                p = np.exp(log_normal_density(app,c['mean'], c['sigma']))*c['weight']
                ax.plot(app,p, linewidth = 0.4)
                ax.set_xlabel('$M\ [M_\\odot]$')
        plt.tight_layout()
        fig.savefig(self.output_components +'/components_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        if self.autocorrelation:
            self.compute_autocorrelation()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_n_clusters+'n_clusters_{0}.pdf'.format(self.e_ID), bbox_inches='tight')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))))
        fig.savefig(self.alpha_folder+'/alpha_{0}.pdf'.format(self.e_ID), bbox_inches='tight')
    
    def compute_autocorrelation(self):
        dx = (self.m_max_plot - self.m_min)/1000.
        square = np.sum(self.median_mf**2*dx)
        autocorrelation = []
        taus = []
        for tau in range(self.n_draws//2):
            autocorrelation.append(np.mean(np.array([np.sum((np.array(self.sample_probs)[:,i] - self.median_mf)*(np.array(self.sample_probs)[:,(i+tau)%self.n_draws]-self.median_mf)*dx) for i in range(self.n_draws)])))
            taus.append(tau + 1)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(taus, autocorrelation, marker = '')
        ax.set_xlabel('$\\tau$ [a.u.]')
        ax.set_ylabel('Autocorrelation')
        plt.savefig(self.output_events+'/autocorrelation.pdf', bbox_inches = 'tight')
        
    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """
        
        
        # reconstructed events
        self.output_events = self.output_folder + '/reconstructed_events/'
        if not os.path.exists(self.output_events):
            os.mkdir(self.output_events)
        if not os.path.exists(self.output_events + '/rec_prob/'):
            os.mkdir(self.output_events + '/rec_prob/')
        self.output_recprob = self.output_events + '/rec_prob/'
        if not os.path.exists(self.output_events + '/n_clusters/'):
            os.mkdir(self.output_events + '/n_clusters/')
        self.output_n_clusters = self.output_events + '/n_clusters/'
        if not os.path.exists(self.output_events + '/events/'):
            os.mkdir(self.output_events + '/events/')
        self.output_pltevents = self.output_events + '/events/'
        if not os.path.exists(self.output_events + '/components/'):
            os.mkdir(self.output_events + '/components/')
        self.output_components = self.output_events + '/components/'
        if not os.path.exists(self.output_events + '/pickle/'):
            os.mkdir(self.output_events + '/pickle/')
        self.output_pickle = self.output_events + '/pickle/'
        if not os.path.exists(self.output_events + '/entropy/'):
            os.mkdir(self.output_events + '/entropy/')
        self.output_entropy = self.output_events + '/entropy/'
        if not os.path.exists(self.output_events + '/alpha/'):
            os.mkdir(self.output_events + '/alpha/')
        self.alpha_folder = self.output_events + '/alpha/'
        self.run_sampling()
        self.postprocess()
        return

class MF_Sampler():
    # inheriting from actor class is not currently supported
    def __init__(self, posterior_functions_events,
                       burnin,
                       n_draws,
                       step,
                       delta_M = 1,
                       alpha0 = 1,
                       m_min = 5,
                       m_max = 50,
                       output_folder = './',
                       verbose = True,
                       initial_cluster_number = 5.,
                       injected_density = None,
                       true_masses = None,
                       diagnostic = False,
                       sigma_max = 5,
                       m_max_plot = 50,
                       autocorrelation = False,
                       n_parallel_threads = 1
                       ):
                       
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.m_min   = m_min
        self.m_max   = m_max
        self.sigma_max = sigma_max
        self.posterior_functions_events = posterior_functions_events
        self.delta_M = delta_M
        self.m_max_plot = m_max_plot
        # DP parameters
        self.alpha0 = alpha0
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        # Output
        self.output_folder = output_folder
        self.mixture_samples = []
        self.n_clusters = []
        self.verbose = verbose
        self.injected_density = injected_density
        self.true_masses = true_masses
        self.diagnostic = diagnostic
        self.autocorrelation = autocorrelation
        self.n_parallel_threads = n_parallel_threads
        self.alpha_samples = []
        
    def initial_state(self):
        self.update_draws()
        assign = [int(a//(len(self.posterior_functions_events)/int(self.icn))) for a in range(len(self.posterior_functions_events))]
        cluster_ids = list(np.arange(int(np.max(assign)+1)))
        state = {
            'cluster_ids_': cluster_ids,
            'data_': self.posterior_draws,
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'Ntot': len(self.posterior_draws),
            'assignment': assign,
            'pi': {cid: self.alpha0 / self.icn for cid in cluster_ids},
            'ev_in_cl': {cid: list(np.where(np.array(assign) == cid)[0]) for cid in cluster_ids},
            'logL_D': {cid: None for cid in cluster_ids}
            }
        for cid in state['cluster_ids_']:
            events = [self.posterior_draws[i] for i in state['ev_in_cl'][cid]]
            n = len(events)
            state['logL_D'][cid] = self.log_numerical_predictive(events, self.m_min, self.m_max, 0.1, self.sigma_max, n)
        state['logL_D']["new"] = self.log_numerical_predictive([], self.m_min, self.m_max, 0.1, self.sigma_max, 0)
        return state
    
    def log_predictive_likelihood(self, data_id, cluster_id, state):
        if cluster_id == "new":
            events = []
            return -np.log(self.m_max-self.m_min), -np.log(self.m_max-self.m_min)
        else:
            events = [self.posterior_draws[i] for i in state['ev_in_cl'][cluster_id]]
        n = len(events)
        events.append(self.posterior_draws[data_id])
        logL_D = state['logL_D'][cluster_id] #denominator
        logL_N = self.log_numerical_predictive(events, self.m_min, self.m_max, 0.1, self.sigma_max, n+1) #numerator
        return logL_N - logL_D, logL_N

    def log_numerical_predictive(self, events, m_min, m_max, sigma_min, sigma_max, n):
        # spezzare il dominio con ray.get()?
        #I, dI = dblquad(integrand, m_min, m_max, gfun = lambda x: sigma_min, hfun = lambda x: sigma_max, args = [np.array(events), m_min, m_max, sigma_min, sigma_max, n])
        I, dI = mp.quad(lambda mu, sigma: np.exp(np.sum([my_logsumexp(np.array([np.log(component['weight']) + mp.npdf(mu, component['mean'], component['sigma'])  for component in ev.values()])) for ev in events])), [m_min, m_max], [sigma_min, sigma_max])
        print(I)
        if (I > 0.0 and np.isfinite(I)):
            return offset + np.log(I)
        else:
            return -np.inf
    
    def cluster_assignment_distribution(self, data_id, state):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster.
        """
        cluster_ids = list(state['ev_in_cl'].keys()) + ['new']
        # can't pickle injected density
        saved_injected_density = self.injected_density
        self.injected_density  = None
        with Pool(self.n_parallel_threads) as p:
            output = p.map(self.compute_score, [[data_id, cid, state] for cid in cluster_ids])
        #output = [self.compute_score([data_id, cid, state]) for cid in cluster_ids]
        scores = {out[0]: out[1] for out in output}
        self.numerators = {out[0]: out[2] for out in output}
        self.injected_density = saved_injected_density
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores
        
    def compute_score(self, args):
        data_id = args[0]
        cid     = args[1]
        state   = args[2]
        score, logL_N = self.log_predictive_likelihood(data_id, cid, state)
        score += self.log_cluster_assign_score(cid, state)
        score = np.exp(score)
        return [cid, score, logL_N]
        
        
    def log_cluster_assign_score(self, cluster_id, state):
        """Log-likelihood that a new point generated will
        be assigned to cluster_id given the current state.
        """
        if cluster_id == "new":
            return np.log(state["alpha_"])
        else:
            if len(state['ev_in_cl'][cluster_id]) == 0:
                return -np.inf
            return np.log(len(state['ev_in_cl'][cluster_id]))

    def create_cluster(self, state):
        state["num_clusters_"] += 1
        cluster_id = max(state['cluster_ids_']) + 1
        state['cluster_ids_'].append(cluster_id)
        state['ev_in_cl'][cluster_id] = []
        return cluster_id

    def destroy_cluster(self, state, cluster_id):
        state["num_clusters_"] -= 1
        state['cluster_ids_'].remove(cluster_id)
        state['ev_in_cl'].pop(cluster_id)
        
    def prune_clusters(self,state):
        for cid in state['cluster_ids_']:
            if len(state['ev_in_cl'][cid]) == 0:
                self.destroy_cluster(state, cid)

    def sample_assignment(self, data_id, state):
        """
        Sample new assignment from marginal distribution.
        If cluster is "`new`", create a new cluster.
        """
        self.numerators = {}
        scores = self.cluster_assignment_distribution(data_id, state).items()
        labels, scores = zip(*scores)
        cid = random.choice(labels, p=scores)
        if cid == "new":
            new_cid = self.create_cluster(state)
            state['logL_D'][int(new_cid)] = self.numerators[cid]
            return new_cid
        else:
            state['logL_D'][int(cid)] = self.numerators[int(cid)]
            return int(cid)

    def update_draws(self):
        draws = []
        for posterior_samples in self.posterior_functions_events:
            draws.append(posterior_samples[random.randint(len(posterior_samples))])
        self.posterior_draws = draws
    
    def drop_from_cluster(self, state, data_id, cid):
        state['ev_in_cl'][cid].remove(data_id)
        events = [self.posterior_draws[i] for i in state['ev_in_cl'][cid]]
        n = len(events)
        state['logL_D'][cid] = self.log_numerical_predictive(events, self.m_min, self.m_max, 0.1, self.sigma_max, n)

    def add_to_cluster(self, state, data_id, cid):
        state['ev_in_cl'][cid].append(data_id)

    def update_alpha(self, state, trimming = 100):
        a_old = state['alpha_']
        n     = state['Ntot']
        K     = len(state['cluster_ids_'])
        for _ in range(trimming):
            a_new = random.RandomState().gamma(1)
            logP_old = gammaln(a_old) - gammaln(a_old + n) + K * np.log(a_old)
            logP_new = gammaln(a_new) - gammaln(a_new + n) + K * np.log(a_new)
            if logP_new - logP_old > np.log(random.uniform()):
                a_old = a_new
        return a_old
    
    def gibbs_step(self, state):
        """
        Collapsed Gibbs sampler for Dirichlet Process Mixture Model
        """
        self.update_draws()
        state['alpha_'] = self.update_alpha(state)
        self.alpha_samples.append(state['alpha_'])
        pairs = zip(state['data_'], state['assignment'])
        for data_id, (datapoint, cid) in enumerate(pairs):
            self.drop_from_cluster(state, data_id, cid)
            self.prune_clusters(state)
            cid = self.sample_assignment(data_id, state)
            self.add_to_cluster(state, data_id, cid)
            state['assignment'][data_id] = cid
        self.n_clusters.append(len(state['cluster_ids_']))
    
    def sample_mixture_parameters(self, state):
        alpha = [len(state['ev_in_cl'][cid]) + state['alpha_'] / state['num_clusters_'] for cid in state['cluster_ids_']]
        weights = stats.dirichlet(alpha).rvs(size=1).flatten()
        components = {}
        for i, cid in enumerate(state['cluster_ids_']):
            events = [self.posterior_draws[j] for j in state['ev_in_cl'][cid]]
            m, s = sample_point(events, self.m_min, self.m_max, 0.1, self.sigma_max, burnin = 1000)
            components[i] = {'mean': m, 'sigma': s, 'weight': weights[i]}
        self.mixture_samples.append(components)
    
    
    def display_config(self):
        print('MCMC Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.mass_samples)))
        print('Concentration parameters:\ngamma0 = {0}'.format(self.alpha0))
        print('Burn-in: {0} samples'.format(self.burnin))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws, self.step))
        print('------------------------')
        return

    def postprocess(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution.
        """
        
        app  = np.linspace(self.m_min, self.m_max_plot, 1000)
        da = app[1]-app[0]
        percentiles = [50, 5,16, 84, 95]
        
        p = {}
        
        fig = plt.figure()
        fig.suptitle('Observed mass function')
        ax  = fig.add_subplot(111)
        if self.true_masses is not None:
            truths = np.genfromtxt(self.true_masses, names = True)
            ax.hist(truths['m'], bins = int(np.sqrt(len(truths['m']))), histtype = 'step', density = True)
        prob = []
        for a in app:
            prob.append([logsumexp([log_normal_density(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) for sample in self.mixture_samples])
        for perc in percentiles:
            p[perc] = np.percentile(prob, perc, axis = 1)
        self.sample_probs = prob
        self.median_mf = np.array(p[50])
        names = ['m'] + [str(perc) for perc in percentiles]
        np.savetxt(self.output_events + '/log_rec_obs_prob_mf.txt', np.array([app, p[50], p[5], p[16], p[84], p[95]]).T, header = ' '.join(names))
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        
        ax.fill_between(app, p[95], p[5], color = 'lightgreen', alpha = 0.5)
        ax.fill_between(app, p[84], p[16], color = 'aqua', alpha = 0.5)
        ax.plot(app, p[50], marker = '', color = 'r')
        if self.injected_density is not None:
            norm = np.sum([self.injected_density(a)*(app[1]-app[0]) for a in app])
            density = np.array([self.injected_density(a)/norm for a in app])
            ax.plot(app, density, color = 'm', marker = '', linewidth = 0.7)
        ax.set_xlabel('$M\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        plt.savefig(self.output_events + '/obs_mass_function.pdf', bbox_inches = 'tight')
        ax.set_yscale('log')
        ax.set_ylim(np.min(p[50]))
        plt.savefig(self.output_events + '/log_obs_mass_function.pdf', bbox_inches = 'tight')
        if self.injected_density is not None:
            self.ppplot(p, app)
        
        name = self.output_events + '/posterior_functions_mf_'
        extension ='.pkl'
        x = 0
        fileName = name + str(x) + extension
        while os.path.exists(fileName):
            x = x + 1
            fileName = name + str(x) + extension
        picklefile = open(fileName, 'wb')
        pickle.dump(self.mixture_samples, picklefile)
        picklefile.close()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_events+'n_clusters_mf.pdf', bbox_inches='tight')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))))
        fig.savefig(self.output_events+'/gamma_mf.pdf', bbox_inches='tight')

        if self.diagnostic:
            fig = plt.figure()
            for i, s in enumerate(self.mixture_samples[:25]):
                ax = fig.add_subplot(5,int(len(self.mixture_samples[:25])/5),i+1)
                app = np.linspace(min(self.mass_samples),max(self.mass_samples),1000)
                for c in s.values():
                    p = np.exp(log_normal_density(app,c['mean'], c['sigma']))*c['weight']
                    ax.plot(app,p, linewidth = 0.4)
                    ax.set_xlabel('$M\ [M_\\odot]$')
            plt.tight_layout()
            fig.savefig(self.output_events +'/components_mf.pdf', bbox_inches = 'tight')
        if self.autocorrelation:
            self.compute_autocorrelation()
    
    def compute_autocorrelation(self):
        dx = (self.m_max_plot - self.m_min)/1000.
        square = np.sum(self.median_mf**2*dx)
        autocorrelation = []
        taus = []
        for tau in range(self.n_draws//2):
            autocorrelation.append(np.mean(np.array([np.sum((np.array(self.sample_probs)[:,i] - self.median_mf)*(np.array(self.sample_probs)[:,(i+tau)%self.n_draws]-self.median_mf)*dx) for i in range(self.n_draws)])))
            taus.append(tau + 1)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(taus, autocorrelation, marker = '')
        ax.set_xlabel('$\\tau$ [a.u.]')
        ax.set_ylabel('Autocorrelation')
        plt.savefig(self.output_events+'/autocorrelation.pdf', bbox_inches = 'tight')

    
    def ppplot(self, p, a):
        f50 = interp1d(a, p[50], bounds_error = False, fill_value = 0)
        cdft  = []
        cdf50 = []
        norm = np.sum([self.injected_density(ai)*(a[1]-a[0]) for ai in a])
        for i in range(len(a)):
            cdft.append(np.sum([self.injected_density(xi)**(a[1]-a[0])/norm for xi in a[:i+1]]))
            cdf50.append(np.sum([f50(xi)**(a[1]-a[0]) for xi in a[:i+1]]))
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        fig.suptitle('PP plot')
        ax.set_xlabel('Simulated f(M)')
        ax.set_ylabel('Reconstructed f(M)')
        ax.plot(np.linspace(0,1, 100), np.linspace(0,1,100), marker = '', linewidth = 0.5, color = 'k')
        ax.plot(cdft, cdf50, ls = '--', marker = '', color = 'r')
        plt.savefig(self.output_events+'PPplot.pdf', bbox_inches = 'tight')
        
        rec_median = np.array([f50(ai) for ai in a])
        inj = np.array([self.injected_density(ai)/norm for ai in a])
        ent = entropy(inj,rec_median)
        print('Relative entropy (Kullback-Leiden divergence): {0} nats'.format(ent))
        np.savetxt(self.output_events + '/relative_entropy.txt', np.array([ent]))
        
    
    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """

        self.run_sampling()
        # reconstructed events
        self.output_events = self.output_folder
        if not os.path.exists(self.output_events):
            os.mkdir(self.output_events)
        self.postprocess()
        return


    def run_sampling(self):
        self.state = self.initial_state()
        for i in range(self.burnin):
            print('\rBURN-IN MF: {0}/{1}'.format(i+1, self.burnin), end = '')
            self.gibbs_step(self.state)
        print('\n', end = '')
        for i in range(self.n_draws):
            print('\rSAMPLING MF: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for _ in range(self.step):
                self.gibbs_step(self.state)
            self.sample_mixture_parameters(self.state)
        print('\n', end = '')
        return
        

def log_normal_density(x, x0, sigma):
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
    
@jit(nopython = True, nogil = True, cache = True)
def log_norm(x, x0, sigma1, sigma2):
    return -((x-x0)**2)/(2*(sigma1**2)) - np.log(np.sqrt(2*np.pi)) - 0.5*np.log(sigma1**2)


def integrand(sigma, mu, events, m_min, m_max, sigma_min, sigma_max, n):
    #logs = ray.get([compute_logsumexp.remote(mu, sigma, ev) for ev in events])
    return mp.exp(np.sum([my_logsumexp(np.array([mp.log(component['weight']) + log_norm(mu, component['mean'], sigma, component['sigma'])  for component in ev.values()])) for ev in events]))
    #return np.exp(np.sum(logs))

#@ray.remote(num_cpus = 4)
#def compute_logsumexp(mu, sigma, event):
#    return my_logsumexp(np.array([np.log(component['weight']) + log_norm(mu, component['mean'], sigma, component['sigma'])  for component in event.values()]))

#@jit(nopython = True, nogil = True, cache = True)
def my_logsumexp(a):
    a_max = a.max()
    tmp = np.array([float(mp.exp(ai - a_max)) for ai in a])
    s = tmp.sum()
    out = mp.log(s)
    out += a_max
    return out

# π ∑ w_k e^(mu - mu_k)^2/sigma
