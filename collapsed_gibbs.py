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

from numba import jit
"""
Implemented as in https://dp.tdhopper.com/collapsed-gibbs/
Multivariate Student-t from http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
"""

@jit(forceobj=True)
def my_student_t(df, t, mu, sigma, dim, sigma_max = 50):


    vals, vecs = np.linalg.eigh(sigma)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    valsinv    = np.minimum(valsinv, sigma_max**2)
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

class StarClusters:
    def __init__(self, catalog,
                       burnin,
                       n_draws,
                       step,
                       alpha0 = 1,
                       L  = 5,
                       k  = 5,
                       nu = 5,
                       output_folder = './',
                       initial_cluster_number = 30,
                       maximum_sigma_cluster = 30.
                       ):
        
        self.catalog = catalog
        try:
            self.dim = np.shape(self.catalog[0])[0]
        except:
            self.dim = 1
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        # DP parameters
        self.alpha0 = alpha0
        # Student-t parameters
        self.L  = (L**2*(len(self.catalog)/initial_cluster_number))*np.identity(self.dim)
        self.k  = k
        self.nu  = nu
        self.mu = np.atleast_2d(np.mean(catalog, axis = 0))
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        self.SuffStat = namedtuple('SuffStat', 'mean cov N')
        # Output
        self.output_folder = output_folder
        self.mixture_samples = []
        self.n_clusters = []
        self.maximum_sigma_cluster = maximum_sigma_cluster
        self.field = []
        
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
            components[cid] = {'mean': m, 'cov': s, 'weight': weights[i], 'N': N}
            
        self.select_field(state, components)
        self.mixture_samples.append(components)
    
    def select_field(self, state, components):
        assign = state['assignment']
        keys = components.keys()
        for key in keys:
            var = components[key]['cov']
            vals, vecs = np.linalg.eigh(var)
            if (vals > self.maximum_sigma_cluster**2).all() or assign.count(key) == 1:
                assign = [x if not x == key else -1 for x in assign]
        self.field.append(assign)
    
    def compute_probability_field(self):
            
        field = np.array(self.field)
        self.p_f = []
        for i in range(len(self.catalog)):
            f = list(field[:,i])
            n_f = f.count(-1)
            tot = len(f)
            self.p_f.append(n_f/tot)
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        c = ax.scatter(self.catalog[:,0], self.catalog[:,1], c = self.p_f, cmap = 'coolwarm', marker = '.', s = 0.3)
        plt.colorbar(c, label = '$p_{field}$')
        plt.savefig(self.output_events + '/field_probability.pdf', bbox_inches = 'tight')
        
    def run_sampling(self):
        state = self.initial_state(self.catalog)
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
        
        fig = plt.figure()
        if self.dim == 2:
            ax  = fig.add_subplot(111)
            c = ax.scatter(self.catalog[:,0], self.catalog[:,1], c = self.last_state['assignment'], marker = '.', s = 0.3)
            plt.colorbar(c)
            plt.savefig(self.output_events + '/cluster_map.pdf', bbox_inches = 'tight')
        if self.dim == 3:
            ax  = fig.add_subplot(111, projection = '3d')
            ax.scatter(self.catalog[:,0], self.catalog[:,1], self.catalog[:,2], c = self.last_state['assignment'], marker = '.')
            plt.savefig(self.output_events + '/cluster_map.pdf', bbox_inches = 'tight')
            
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
        
        self.compute_probability_field()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(self.output_folder+'n_clusters.pdf', bbox_inches='tight')
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
