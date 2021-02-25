import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings

from collections import namedtuple, Counter
from scipy import stats
from numpy import random
from scipy.stats import t as student_t
from scipy.special import logsumexp, betaln, gammaln
from scipy.interpolate import interp1d

from time import perf_counter

from numba import jit
import ray
from ray.util import ActorPool

"""
Implemented as in https://dp.tdhopper.com/collapsed-gibbs/
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
                       mass_chain_settings = [100,1], # burnin masses, step masses
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
                       injected_density = None
                       ):
        
        self.events = events
        self.burnin_mf, self.n_draws_mf, self.step_mf = samp_settings
        if samp_settings_ev is not None:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings_ev
        else:
            self.burnin_ev, self.n_draws_ev, self.step_ev = samp_settings
        self.burnin_masses, self.step_masses = mass_chain_settings
        self.m_min   = m_min
        self.m_max   = m_max
        # DP
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        # student-t
        self.a_mf, self.b_mf, self.V_mf = hyperpars
        if hyperpars_ev is not None:
            self.a_ev, self.b_ev, self.V_ev = hyperpars_ev
        else:
            self.a_ev, self.b_ev, self.V_ev = hyperpars
        # miscellanea
        self.output_folder = output_folder
        self.icn = initial_cluster_number
        self.event_samplers = []
        self.delta_M = np.std(events, axis = 1)
        self.verbose = verbose
        self.process_events = process_events
        self.n_parallel_threads = n_parallel_threads
        self.injected_density = injected_density
        self.output_recprob = self.output_folder + '/reconstructed_events/rec_prob/'
    
    def initialise_samplers(self, marker):
        event_samplers = []
        for i, ev in enumerate(self.events[marker:marker+self.n_parallel_threads]):
            event_samplers.append(Sampler_SE.remote(
                                            ev,
                                            marker + i+1,
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
                                            self.icn
                                            ))
        return event_samplers
        
    def run_event_sampling(self):
        i = 0
        for n in range(int(len(self.events)/self.n_parallel_threads)+1):
            tasks = self.initialise_samplers(n*self.n_parallel_threads)
            pool = ActorPool(tasks)
            for s in pool.map_unordered(lambda a, v: a.run.remote(), range(len(tasks))):
                i += 1
                print('\rProcessed {0}/{1} events\r'.format(i, len(self.events)), end = '')
        return
    
    def load_mixtures(self):
        print('Loading mixtures...')
        self.log_mass_posteriors = []
        prob_files = [self.output_recprob+f for f in os.listdir(self.output_recprob) if f.startswith('log_rec_prob_')]
        prob_files.sort(key = natural_keys)
        for prob in prob_files:
            rec_prob = np.genfromtxt(prob)
            self.log_mass_posteriors.append(interp1d(rec_prob[:,0], rec_prob[:,1], bounds_error = False, fill_value = 0.))
    
    def display_config(self):
        print('Collapsed Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.events)))
        print('Concentration parameters:\nalpha0 = {0}\tgamma0 = {1}'.format(self.alpha0, self.gamma0))
        print('Burn-in: {0} samples'.format(self.burnin_mf))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws_mf, self.step_mf))
        print('------------------------')
        return
    
    def initialise_mt_samples(self):
        self.mt = np.mean(self.events, axis = 1)
    
    def run_mass_function_sampling(self):
        self.load_mixtures()
        self.initialise_mt_samples()
        sorted = sort_matrix(a = [self.mt, self.log_mass_posteriors], axis = 0)
        self.mt = sorted[0]
        self.log_mass_posteriors = sorted[1]
        self.mf_folder = self.output_folder+'/mass_function/'
        if not os.path.exists(self.mf_folder):
            os.mkdir(self.mf_folder)
            
        sampler = MF_Sampler(self.mt,
                       self.log_mass_posteriors,
                       self.burnin_mf,
                       self.n_draws_mf,
                       self.step_mf,
                       delta_M = self.delta_M,
                       alpha0 = self.gamma0,
                       #eventualmente differenziare iperparametri interni ed esterni
                       #ora è così solo per finalità di test
                       b = self.b_mf,
                       a = self.a_mf,
                       V = self.V_mf,
                       m_min = self.m_min,
                       m_max = self.m_max,
                       verbose = self.verbose,
                       output_folder = self.mf_folder,
                       initial_cluster_number = self.icn,
                       injected_density = self.injected_density,
                       burnin_masses = self.burnin_masses,
                       step_masses = self.step_masses
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
        
ray.init(ignore_reinit_error=True, log_to_driver=False)

@jit(forceobj=True)
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
                       initial_cluster_number = 5.
                       ):
        
        self.mass_samples  = mass_samples
        self.e_ID    = event_id
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.m_min   = m_min
        self.m_max   = m_max
        # DP parameters
        self.alpha0 = alpha0
        # Student-t parameters
        self.b  = (b**2)*len(mass_samples)/initial_cluster_number
        self.a  = len(mass_samples)/(initial_cluster_number/2.)
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
        
    def initial_state(self, samples):
        cluster_ids = list(np.arange(int(self.icn)))
        state = {
            'cluster_ids_': cluster_ids,
            'data_': np.sort(samples),
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'hyperparameters_': {
                "b": self.b,
                "a": self.a,
                "V": self.V,
                "mu": self.mu
                },
            'suffstats': {cid: None for cid in cluster_ids},
            'assignment': [int((a - a%(len(samples)/self.icn))/(len(samples)/self.icn)) for a in range(len(samples))],
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
        cid = random.choice(labels, p=scores)
        if cid == "new":
            return self.create_cluster(state)
        else:
            return int(cid)

    def gibbs_step(self, state):
        """
        Collapsed Gibbs sampler for Dirichlet Process Mixture Model
        """
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
        weights = stats.dirichlet(alpha).rvs(size=1).flatten()
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

    def plot_samples(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution.
        """
        
        app  = np.linspace(self.m_min, self.m_max, 1000)
        percentiles = [5,16, 50, 84, 95]
        
        p = {}
        
        fig = plt.figure()
        fig.suptitle('Event {0}'.format(self.e_ID))
        ax  = fig.add_subplot(111)
        ax.hist(self.mass_samples, bins = int(np.sqrt(len(self.mass_samples))), histtype = 'step', density = True)
        prob = []
        for a in app:
            prob.append([logsumexp([log_normal_density(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) for sample in self.mixture_samples])
        p[50] = np.percentile(prob, 50, axis = 1)
        np.savetxt(self.output_recprob + '/log_rec_prob_{0}.txt'.format(self.e_ID), np.array([app, p[50]]).T)
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        
        ax.fill_between(app, p[95], p[5], color = 'lightgreen', alpha = 0.5)
        ax.fill_between(app, p[84], p[16], color = 'aqua', alpha = 0.5)
        ax.plot(app, p[50], marker = '', color = 'r')
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        
        plt.savefig(self.output_pltevents + '/event_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        fig = plt.figure()
        for i, s in enumerate(self.mixture_samples[:25]):
            ax = fig.add_subplot(5,int(len(self.mixture_samples[:25])/5),i+1)
            app = np.linspace(min(self.mass_samples),max(self.mass_samples),1000)
            for c in s.values():
                p = np.exp(log_normal_density(app,c['mean'], c['sigma']))*c['weight']
                ax.plot(app,p, linewidth = 0.4)
                ax.set_xlabel('$M_1\ [M_\\odot]$')
        plt.tight_layout()
        fig.savefig(self.output_components +'/components_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        
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
        self.run_sampling()
        self.plot_samples()
        
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_n_clusters+'n_clusters_{0}.pdf'.format(self.e_ID), bbox_inches='tight')
        return

class MF_Sampler():
    # inheriting from actor class is not currently supported
    def __init__(self, mass_samples,
                       log_mass_posteriors,
                       burnin,
                       n_draws,
                       step,
                       delta_M = 1,
                       alpha0 = 1,
                       b = 5,
                       a = 3,
                       V = 1/4.,
                       m_min = 5,
                       m_max = 50,
                       output_folder = './',
                       verbose = True,
                       initial_cluster_number = 5.,
                       injected_density = None,
                       burnin_masses = 0,
                       step_masses = 1
                       ):
                       
        self.mass_samples  = mass_samples
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.burnin_masses = burnin_masses
        self.step_masses = step_masses
        self.m_min   = m_min
        self.m_max   = m_max
        self.log_mass_posteriors = log_mass_posteriors
        self.delta_M = delta_M
        # DP parameters
        self.alpha0 = alpha0
        # Student-t parameters
        self.b  = (b**2)*len(mass_samples)/initial_cluster_number
        self.a  = len(mass_samples)/(initial_cluster_number/2.)
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
        self.injected_density = injected_density
        
    def initial_state(self, samples):
        cluster_ids = list(np.arange(int(self.icn)))
        state = {
            'cluster_ids_': cluster_ids,
            'data_': samples,
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'hyperparameters_': {
                "b": self.b,
                "a": self.a,
                "V": self.V,
                "mu": self.mu
                },
            'suffstats': {cid: None for cid in cluster_ids},
            'assignment': [int((a - a%(len(samples)/self.icn))/(len(samples)/self.icn)) for a in range(len(samples))],
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
            # print(var, points_in_cluster)
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
        #if b_n*(1+V_n)/a_n < 0:
            # print(x, mean, sigma, N)
            # print(self.mass_samples)
        t_sigma = np.sqrt(b_n*(1+V_n)/a_n)
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
        # print((ss.N*(ss.var + ss.mean**2) - x**2)/(ss.N-1), mean**2, var, ss.N-1, mean, np.sqrt(ss.var), x)
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
        cid = random.choice(labels, p=scores)
        if cid == "new":
            return self.create_cluster(state)
        else:
            return int(cid)

    def gibbs_step(self, state):
        """
        Collapsed Gibbs sampler for Dirichlet Process Mixture Model
        """
        pairs = zip(state['data_'], state['assignment'])
        for data_id, (datapoint, cid) in enumerate(pairs):
            # print(cid)
            state['suffstats'][cid] = self.remove_datapoint_from_suffstats(datapoint, state['suffstats'][cid])
            self.prune_clusters(state)
            cid = self.sample_assignment(data_id, state)
            state['assignment'][data_id] = cid
            state['suffstats'][cid] = self.add_datapoint_to_suffstats(state['data_'][data_id], state['suffstats'][cid])
        self.n_clusters.append(len(state['cluster_ids_']))
    
    def sample_mixture_parameters(self, state):
        ss = state['suffstats']
        alpha = [ss[cid].N + state['alpha_'] / state['num_clusters_'] for cid in state['cluster_ids_']]
        weights = stats.dirichlet(alpha).rvs(size=1).flatten()
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
    
    
    def display_config(self):
        print('MCMC Gibbs sampler')
        print('------------------------')
        print('Loaded {0} events'.format(len(self.mass_samples)))
        print('Concentration parameters:\ngamma0 = {0}'.format(self.alpha0))
        print('Burn-in: {0} samples'.format(self.burnin))
        print('Samples: {0} - 1 every {1}'.format(self.n_draws, self.step))
        print('------------------------')
        return

    def plot_samples(self):
        """
        Plots samples [x] for each event in separate plots along with inferred distribution.
        """
        
        app  = np.linspace(self.m_min, self.m_max, 1000)
        percentiles = [5,16, 50, 84, 95]
        
        p = {}
        
        fig = plt.figure()
        fig.suptitle('Mass function')
        ax  = fig.add_subplot(111)
        ax.hist(self.mass_samples, bins = int(np.sqrt(len(self.mass_samples))), histtype = 'step', density = True)
        prob = []
        for a in app:
            prob.append([logsumexp([log_normal_density(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) for sample in self.mixture_samples])
        p[50] = np.percentile(prob, 50, axis = 1)
        np.savetxt(self.output_events + '/log_rec_prob_mf.txt', np.array([app, p[50]]).T)
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        
        ax.fill_between(app, p[95], p[5], color = 'lightgreen', alpha = 0.5)
        ax.fill_between(app, p[84], p[16], color = 'aqua', alpha = 0.5)
        ax.plot(app, p[50], marker = '', color = 'r')
        if self.injected_density is not None:
            ax.plot(app, self.injected_density(app), color = 'k', marker = '')
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        plt.savefig(self.output_events + '/mass_function.pdf', bbox_inches = 'tight')
        fig = plt.figure()
        for i, s in enumerate(self.mixture_samples[:25]):
            ax = fig.add_subplot(5,int(len(self.mixture_samples[:25])/5),i+1)
            app = np.linspace(min(self.mass_samples),max(self.mass_samples),1000)
            for c in s.values():
                p = np.exp(log_normal_density(app,c['mean'], c['sigma']))*c['weight']
                ax.plot(app,p, linewidth = 0.4)
                ax.set_xlabel('$M_1\ [M_\\odot]$')
        plt.tight_layout()
        fig.savefig(self.output_events +'/components_mf.pdf', bbox_inches = 'tight')
        
    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """
        
        self.run_sampling()
        # reconstructed events
        self.output_events = self.output_folder
        if not os.path.exists(self.output_events):
            os.mkdir(self.output_events)
        self.plot_samples()
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_events+'n_clusters_mf.pdf', bbox_inches='tight')
        return


    def update_mass_posteriors(self, state):
        # Parallelizzabile
        for _ in range(self.step_masses):
            for index in range(len(self.log_mass_posteriors)):
                self.update_single_posterior(index)
        state['data_'] = self.mass_samples
    
    def update_single_posterior(self, e_index):
        M_old = self.mass_samples[e_index]
        M_new = draw_mass(M_old, self.delta_M[e_index])
        if not self.m_min < M_new < self.m_max:
            return
        p_old = self.log_mass_posteriors[e_index](M_old)
        p_new = self.log_mass_posteriors[e_index](M_new)
        
        if p_new - p_old > np.log(random.uniform()):
            self.mass_samples[e_index] = M_new
        return
    
    def run_sampling(self):
        self.check_masses = [[] for _ in range(len(self.mass_samples))]
        state = self.initial_state(self.mass_samples)
        for i in range(self.burnin_masses):
            print('\rTERMALIZING MASS SAMPLES: {0}/{1}'.format(i+1, self.burnin_masses), end = '')
            self.update_mass_posteriors(state)
        state = self.initial_state(self.mass_samples)
        print('\n', end = '')
        for i in range(self.burnin):
            print('\rBURN-IN MF: {0}/{1}'.format(i+1, self.burnin), end = '')
            self.update_mass_posteriors(state)
            self.update_suffstats(state)
            self.gibbs_step(state)
        print('\n', end = '')
        for i in range(self.n_draws):
            print('\rSAMPLING MF: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for _ in range(self.step):
                self.update_mass_posteriors(state)
                self.update_suffstats(state)
                self.gibbs_step(state)
            self.sample_mixture_parameters(state)
        print('\n', end = '')
        return
        
@jit()
def draw_mass(Mold, delta_M):
    return Mold + delta_M * random.uniform(-1,1)


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

