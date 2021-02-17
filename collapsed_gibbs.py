import numpy as np
import matplotlib.pyplot as plt
import os

from collections import namedtuple, Counter
from scipy import stats
from numpy import random
from scipy.stats import t as student_t
from scipy.special import logsumexp, betaln

from numba import jit
import ray
from ray.util import ActorPool

"""
Implemented as in https://dp.tdhopper.com/collapsed-gibbs/
"""

class CGSampler:
                
    def __init__(self, events,
                       burnin,
                       n_draws,
                       step,
                       alpha0 = 1,
                       gamma0 = 1,
                       b = 5,
                       a = 3,
                       V = 1/4.,
                       m_min = 5,
                       m_max = 50,
                       output_folder = './',
                       initial_cluster_number = 5.
                       ):
        
        self.events = events
        self.burnin = burnin
        self.n_draws = n_draws
        self.step = step
        self.m_min   = m_min
        self.m_max   = m_max
        # DP
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        # student-t
        self.a = a
        self.b = b
        self.V = V
        # miscellanea
        self.output_folder = output_folder
        self.icn = initial_cluster_number
        self.event_samplers = []
    
    def initialise_samplers(self):
        for i, ev in enumerate(self.events):
            self.event_samplers.append(Sampler_SE.remote(
                                            ev,
                                            i+1,
                                            self.burnin,
                                            self.n_draws,
                                            self.step,
                                            self.alpha0,
                                            self.b,
                                            self.a,
                                            self.V,
                                            self.m_min,
                                            self.m_max,
                                            self.output_folder,
                                            self.icn
                                            ))
        return
        
    def run_event_sampling(self):
        self.initialise_samplers()
        tasks = [s for s in self.event_samplers]
        pool = ActorPool(tasks)
        for s in pool.map_unordered(lambda a, v: a.run.remote(), range(len(tasks))):
            pass
        
        
ray.init(ignore_reinit_error=True)
    
@jit()
def my_student_t(df, t):
    b = betaln(0.5, df/2.)
    return -0.5*np.log(df)-b-((df+1)*0.5)*np.log1p(t*t/df)
    
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
            print('\rBURN-IN: {0}/{1}'.format(i+1, self.burnin), end = '')
            self.gibbs_step(state)
        print('\n', end = '')
        for i in range(self.n_draws):
            print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for _ in range(self.step):
                self.gibbs_step(state)
            self.sample_mixture_parameters(state)
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
        for perc in percentiles:
            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
        np.savetxt(self.output_events + '/rec_prob_{0}.txt'.format(self.e_ID), np.array(p[50]))
        
        ax.fill_between(app, p[95], p[5], color = 'lightgreen', alpha = 0.5)
        ax.fill_between(app, p[84], p[16], color = 'aqua', alpha = 0.5)
        ax.plot(app, p[50], marker = '', color = 'r')
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        ax.set_ylabel('$p(M)$')
        plt.savefig(self.output_events + '/event_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        fig = plt.figure()
        for i, s in enumerate(self.mixture_samples[:25]):
            ax = fig.add_subplot(5,len(self.mixture_samples[:25])/5,i+1)
            app = np.linspace(min(self.mass_samples),max(self.mass_samples),1000)
            for c in s.values():
                p = np.exp(log_normal_density(app,c['mean'], c['sigma']))*c['weight']
                ax.plot(app,p, linewidth = 0.4)
                ax.set_xlabel('$M_1\ [M_\\odot]$')
        plt.tight_layout()
        fig.savefig(self.output_events +'/components_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        
    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """
        
        flags = []
        print(1)
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
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_folder+'n_clusters_{0}.pdf'.format(self.e_ID), bbox_inches='tight')
        return

#@jit(nopython = True)
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
