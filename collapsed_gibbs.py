import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import pickle

from collections import namedtuple, Counter
from scipy import stats
from numpy import random
from scipy.stats import multivariate_t as student_t
from scipy.special import logsumexp, gammaln
from scipy.stats import multivariate_normal as mn
from numpy.linalg import det, inv

import ray
from ray.util import ActorPool
from numba import jit
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
                       L = 10**-2,
                       k = 1,
                       nu = 3,
                       m_min = 5,
                       m_max = 60,
                       output_folder = './',
                       initial_cluster_number = 10,
                       min_cluster_occupation = 10
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
        self.L = L
        self.k = k
        self.nu = nu
        # miscellanea
        self.output_folder = output_folder
        self.icn = initial_cluster_number
        self.event_samplers = []
        self.min_cluster_occupation = min_cluster_occupation
    
    def initialise_samplers(self):
        for i, ev in enumerate(self.events):
            self.event_samplers.append(Sampler_SE.remote(
                                            ev,
                                            i+1,
                                            self.burnin,
                                            self.n_draws,
                                            self.step,
                                            self.alpha0,
                                            self.L,
                                            self.k,
                                            self.nu,
                                            self.m_min,
                                            self.m_max,
                                            self.output_folder,
                                            self.icn,
                                            self.min_cluster_occupation
                                            ))
        return
        
    def run_event_sampling(self):
        self.initialise_samplers()
        tasks = [s for s in self.event_samplers]
        pool = ActorPool(tasks)
        for s in pool.map_unordered(lambda a, v: a.run.remote(), range(len(tasks))):
            pass
        
        
ray.init(ignore_reinit_error=True)
    
"""
http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
"""
@jit()
def my_student_t(df, t, mu, sigma, dim):

    vals, vecs = np.linalg.eigh(sigma)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = t - mu
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    x = 0.5 * (df + dim)
    A = gammaln(x)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -x * np.log1p((1./df) * maha)

    return float(A - B - C - D + E)

@ray.remote
class Sampler_SE:
    def __init__(self, mass_samples,
                       event_id,
                       burnin,
                       n_draws,
                       step,
                       alpha0 = 1,
                       L  = 5,
                       k  = 5,
                       nu = 5,
                       m_min = 5,
                       m_max = 50,
                       output_folder = './',
                       initial_cluster_number = 10,
                       min_cluster_occupation = 0
                       ):
        
        self.mass_samples  = mass_samples
        try:
            self.dim = np.shape(self.mass_samples[0])[0]
        except:
            self.dim = 1
        self.e_ID    = event_id
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.m_min   = m_min
        self.m_max   = m_max
        # DP parameters
        self.alpha0 = alpha0
        # Student-t parameters
        self.L  = (L**2*(len(self.mass_samples)/initial_cluster_number))*np.identity(self.dim)
        self.k  = k
        self.nu  = nu
        self.mu = np.atleast_2d(np.mean(mass_samples, axis = 0))
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        self.SuffStat = namedtuple('SuffStat', 'mean cov N')
        # Output
        self.output_folder = output_folder
        self.mixture_samples = []
        self.n_clusters = []
        self.min_cluster_occupation = min_cluster_occupation
        
    def initial_state(self, samples):
        cluster_ids = list(np.arange(int(self.icn)))
        assig = np.zeros(len(samples))
        for i in range(int(self.icn)):
            assig[i*(int(len(samples)/self.icn)+1):(i+1)*(int(len(samples)/self.icn)+1)] = i
        state = {
            'cluster_ids_': cluster_ids,
            'data_': samples,
            'num_clusters_': int(self.icn),
            'alpha_': self.alpha0,
            'hyperparameters_': {
                "L": self.L,
                "k": self.k,
                "nu": self.nu,
                "mu": self.mu
                },
            'suffstats': {cid: None for cid in cluster_ids},
            'assignment': list(assig),
            'pi': {cid: self.alpha0 / self.icn for cid in cluster_ids},
            }
        print(state['assignment'])
        self.update_suffstats(state)
        return state
    
    def update_suffstats(self, state):
        for cluster_id, N in Counter(state['assignment']).items():

            points_in_cluster = [x for x, cid in zip(state['data_'], state['assignment']) if cid == cluster_id]
            mean = np.atleast_2d(np.array(points_in_cluster).mean(axis = 0))
            cov  = np.cov(np.array(points_in_cluster), rowvar = False)
            M    = len(points_in_cluster)
            state['suffstats'][cluster_id] = self.SuffStat(mean, cov, M)
    
    def log_predictive_likelihood(self, data_id, cluster_id, state):
        
        if cluster_id == "new":
            ss = self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0)
        else:
            ss  = state['suffstats'][cluster_id]
            
        x = state['data_'][data_id]
        mean = ss.mean
        S    = ss.cov
        N    = ss.N
        # Update hyperparameters
        k_n  = state['hyperparameters_']["k"] + N
        mu_n = np.atleast_2d((state['hyperparameters_']["mu"]*state['hyperparameters_']["k"] + N*mean)/k_n)
        nu_n = state['hyperparameters_']["nu"] + N
        L_n  = state['hyperparameters_']["L"]*state['hyperparameters_']["k"] + S*N + state['hyperparameters_']["k"]*N*np.matmul((mean - state['hyperparameters_']["mu"]).T, (mean - state['hyperparameters_']["mu"]))/k_n
        # Update t-parameters
        t_df    = nu_n - self.dim + 1
        t_shape = L_n*(k_n+1)/(k_n*t_df)
        # Compute logLikelihood
        logL = my_student_t(df = t_df, t = np.atleast_2d(x), mu = mu_n, sigma = t_shape, dim = self.dim)
        return logL

    def add_datapoint_to_suffstats(self, x, ss):
        x = np.atleast_2d(x)
        mean = (ss.mean*(ss.N)+x)/(ss.N+1)
        cov  = (ss.N*(ss.cov + np.matmul(ss.mean.T, ss.mean)) + np.matmul(x.T, x))/(ss.N+1) - np.matmul(mean.T, mean)
        return self.SuffStat(mean, cov, ss.N+1)


    def remove_datapoint_from_suffstats(self, x, ss):
        x = np.atleast_2d(x)
        if ss.N == 1:
            return(self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0))
        mean = (ss.mean*(ss.N)-x)/(ss.N-1)
        cov  = (ss.N*(ss.cov + np.matmul(ss.mean.T, ss.mean)) - np.matmul(x.T, x))/(ss.N-1) - np.matmul(mean.T, mean)
        return self.SuffStat(mean, cov, ss.N-1)
    
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
        state['suffstats'][cluster_id] = self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0)
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
            S = ss[cid].cov
            N     = ss[cid].N
            k_n  = state['hyperparameters_']["k"] + N
            mu_n = np.atleast_2d((state['hyperparameters_']["mu"]*state['hyperparameters_']["k"] + N*mean)/k_n)
            nu_n = state['hyperparameters_']["nu"] + N
            L_n  = state['hyperparameters_']["L"] + S*N + state['hyperparameters_']["k"]*N*np.matmul((mean - state['hyperparameters_']["mu"]).T, (mean - state['hyperparameters_']["mu"]))/k_n
            # Update t-parameters
            s = stats.invwishart(df = nu_n, scale = L_n).rvs()
            t_df    = nu_n - self.dim + 1
            t_shape = L_n*(k_n+1)/(k_n*t_df)
            m = student_t(df = t_df, loc = mu_n.flatten(), shape = t_shape).rvs()
            components[i] = {'mean': m, 'cov': s, 'weight': weights[i], 'N': N}
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
        self.last_state = state
        print('\n', end = '')
        return
    
    def save_mixture_samples(self):
        with open(self.output_folder+ '/mixture_samples.pkl', 'wb') as f:
            pickle.dump(self.mixture_samples, f, pickle.HIGHEST_PROTOCOL)
        samples = {}
        for sample in self.mixture_samples:
            for cluster, key in zip(sample.values(), sample.keys()):
                if not key in samples.keys():
                    samples[key] = []
                if cluster['N'] > self.min_cluster_occupation:
                    l = list(cluster['mean']) + [item for row in cluster['cov'] for item in row]
                    l.append(cluster['N'])
                    samples[key].append(l)
        cluster_folder = self.output_folder+'/clusters/'
        if not os.path.exists(cluster_folder):
            os.mkdir(cluster_folder)
        for i, cluster in enumerate(samples.values()):
            if cluster: # prune empty clusters
                cluster = np.array(cluster)
                np.savetxt(cluster_folder + 'cluster_{0}.txt'.format(i), cluster)
    
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
#        ax.hist(self.mass_samples, bins = int(np.sqrt(len(self.mass_samples))), histtype = 'step', density = True)
#        prob = []
#        for a in app:
#            prob.append([logsumexp([log_normal_density(a, component['mean'], component['sigma']) for component in sample.values()], b = [component['weight'] for component in sample.values()]) for sample in self.mixture_samples])
#        for perc in percentiles:
#            p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
#        np.savetxt(self.output_events + '/rec_prob_{0}.txt'.format(self.e_ID), np.array(p[50]))
#
#        ax.fill_between(app, p[95], p[5], color = 'lightgreen', alpha = 0.5)
#        ax.fill_between(app, p[84], p[16], color = 'aqua', alpha = 0.5)
#        ax.plot(app, p[50], marker = '', color = 'r')
#        ax.set_xlabel('$M_1\ [M_\\odot]$')
#        ax.set_ylabel('$p(M)$')
        if self.dim == 2:
            ax  = fig.add_subplot(111)
            ax.scatter(self.mass_samples[:,0], self.mass_samples[:,1], c = self.last_state['assignment'], marker = '.')
            plt.savefig(self.output_events + '/event_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        elif self.dim == 3:
            ax  = fig.add_subplot(111, projection = '3d')
            ax.scatter(self.mass_samples[:,0], self.mass_samples[:,1], self.mass_samples[:,2], c = self.last_state['assignment'], marker = '.')
            plt.savefig(self.output_events + '/event_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
        elif self.dim == 1:
            plot_clusters(self.last_state, self.output_events + '/event_{0}.pdf'.format(self.e_ID))
#        fig = plt.figure()
#        for i, s in enumerate(self.mixture_samples[:25]):
#            ax = fig.add_subplot(5,len(self.mixture_samples[:25])/5,i+1)
#            app = np.linspace(min(self.mass_samples),max(self.mass_samples),1000)
#            for c in s.values():
#                p = np.exp(log_normal_density(app,c['mean'], c['sigma']))*c['weight']
#                ax.plot(app,p, linewidth = 0.4)
#                ax.set_xlabel('$M_1\ [M_\\odot]$')
#        plt.tight_layout()
#        fig.savefig(self.output_events +'/components_{0}.pdf'.format(self.e_ID), bbox_inches = 'tight')
    
    def run(self):
        """
        Runs sampler, saves samples and produces output plots.
        """
        
        flags = []
        self.run_sampling()
        # reconstructed events
        self.output_events = self.output_folder + '/reconstructed_events'
        if not os.path.exists(self.output_events):
            os.mkdir(self.output_events)
        self.save_mixture_samples()
        if self.dim < 4:
            self.plot_samples()
        self.output_samples_folder = self.output_folder + '/posterior_samples/'
        if not os.path.exists(self.output_samples_folder):
            os.mkdir(self.output_samples_folder)
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_folder+'n_clusters_{0}.pdf'.format(self.e_ID), bbox_inches='tight')
        return

@jit(nopython = True)
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

        
        
def plot_clusters(state, file):
    gby = pd.DataFrame({
            'data': state['data_'],
            'assignment': state['assignment']}
        ).groupby(by='assignment')['data']
    hist_data = [gby.get_group(cid).tolist()
                 for cid in gby.groups.keys()]
    plt.hist(hist_data,
             bins=20,
             histtype='stepfilled', alpha=.5 )
    plt.savefig(file, bbox_inches = 'tight')
