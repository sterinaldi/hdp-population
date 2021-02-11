import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from collections import namedtuple, Counter
from scipy import stats
from numpy import random
from scipy.stats import t as student_t
from scipy.special import logsumexp

import warnings
warnings.filterwarnings("error")

"""
Implemented as in https://dp.tdhopper.com/collapsed-gibbs/
"""

class CGSampler:

    def __init__(self, events,
                       burnin,
                       n_draws,
                       step,
                       mass_b = [5,50],
                       alpha0 = 1,
                       gamma0 = 1,
                       beta = 5,
                       alpha = 5,
                       k = 0.01,
                       output_folder = './',
                       initial_cluster_number = 2.
                       ):
        
        self.events  = events
        self.burnin  = burnin
        self.n_draws = n_draws
        self.step    = step
        self.m_min = min(mass_b)
        self.m_max = max(mass_b)
        # DP parameters
        self.alpha0 = alpha0
        self.gamma0 = gamma0
        # Student-t parameters
        self.beta  = beta
        self.alpha = alpha
        self.k     = k
        self.mu    = np.mean(self.events, axis = 1)
        # Miscellanea
        self.icn    = initial_cluster_number
        self.states = []
        self.SuffStat = namedtuple('SuffStat', 'mean var N')
        # Output
        self.output_folder = output_folder
        self.internal_mixture_samples = [[] for _ in self.events]
        
    def initial_state(self, samples, mu):
        cluster_ids = list(np.arange(int(self.icn)))
        state = {
            'cluster_ids_': cluster_ids,
            'data_': samples,
            'num_clusters_': self.icn,
            'alpha_': self.alpha0,
            'hyperparameters_': {
                "beta": self.beta,
                "alpha": self.alpha,
                "k": self.k,
                "mu": mu
                },
            'suffstats': {cid: None for cid in cluster_ids},
            'assignment': [random.choice(cluster_ids) for _ in samples],
            'pi': {cid: self.alpha / self.icn for cid in cluster_ids},
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
        beta_n  = state['hyperparameters_']["beta"] + sigma*N/2. + state['hyperparameters_']["k"]*N*(mean-state['hyperparameters_']["mu"])**2./(2*(state['hyperparameters_']["k"]+N))
        alpha_n = state['hyperparameters_']["alpha"] + N/2
        k_n     = state['hyperparameters_']["k"] + N
        mu_n    = (state['hyperparameters_']["k"]*state['hyperparameters_']["mu"] + N*mean)/(state['hyperparameters_']["k"] + N)
        # Update t-parameters
        t_sigma = np.sqrt((beta_n*(k_n+1))/(alpha_n*k_n))
        t_x     = (x - mu_n)/t_sigma
        # Compute logLikelihood
        logL = student_t(2*alpha_n).logpdf(t_x)
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
        scores = {cid: score for cid, score in scores.items()}
        normalization = -logsumexp(list(scores.values()))
        scores = {cid: np.exp(score+normalization) for cid, score in scores.items()}
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
    
    def sample_mixture_parameters(self, state, event_id):
        ss = state['suffstats']
        alpha = [ss[cid].N + state['alpha_'] / state['num_clusters_'] for cid in state['cluster_ids_']]
        weights = stats.dirichlet(alpha).rvs(size=1).flatten()
        components = {}
        for i, cid in enumerate(state['cluster_ids_']):
            mean = ss[cid].mean
            sigma = ss[cid].var
            N     = ss[cid].N
            beta_n  = state['hyperparameters_']["beta"] + sigma*N/2. + state['hyperparameters_']["k"]*N*(mean-state['hyperparameters_']["mu"])**2/(2*(state['hyperparameters_']["k"]+N))
            alpha_n = state['hyperparameters_']["alpha"] + N/2
            k_n     = state['hyperparameters_']["k"] + N
            mu_n    = (state['hyperparameters_']["k"]*state['hyperparameters_']["mu"] + N*mean)/(state['hyperparameters_']["k"] + N)
            l = stats.gamma(alpha_n, scale = 1/beta_n).rvs()
            m = stats.norm(mu_n, 1/((k_n+N)*l)).rvs()
            s = np.sqrt(1/l)
            components[i] = {'mean': m, 'sigma': s, 'weight': weights[i]}
        self.internal_mixture_samples[event_id].append(components)
    
    def run_sampling(self):
        for event_id in range(len(self.events)):
            state = self.initial_state(self.events[event_id], self.mu[event_id])
            for i in range(self.burnin):
                print('\rBURN-IN: {0}/{1}'.format(i+1, self.burnin), end = '')
                self.gibbs_step(state)
            print('\n', end = '')
            for i in range(self.n_draws):
                print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws), end = '')
                for _ in range(self.step):
                    self.gibbs_step(state)
                self.sample_mixture_parameters(state, event_id)
            print('\n', end = '')
    
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
        
        for samples, dists, i in zip(self.events, self.internal_mixture_samples, range(len(self.events))):
            fig = plt.figure()
            fig.suptitle('Event {0}'.format(i+1))
            ax  = fig.add_subplot(111)
            ax.hist(samples, bins = int(np.sqrt(len(samples))), histtype = 'step', density = True)
#            finalprob = []
#            for s in self.internal_posterior_samples[i]:
#                probs = np.array([self.log_normal_density(app, component['mean'], component['sigma']) for component in s])
#                ws = np.atleast_2d(np.array([component['weight'] for component in s]))
#                finalprob.append(logsumexp(probs, b = ws.T, axis = 0))
            prob = []
            for a in app:
                prob.append([logsumexp([self.log_normal_density(a, component['mean'], component['sigma']) for component in dist.values()], b = [component['weight'] for component in dist.values()]) for dist in dists])
                
            for perc in percentiles:
                p[perc] = np.exp(np.percentile(prob, perc, axis = 1))
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

def plot_clusters(state):
    gby = pd.DataFrame({
            'data': state['data_'],
            'assignment': state['assignment']}
        ).groupby(by='assignment')['data']
    hist_data = [gby.get_group(cid).tolist()
                 for cid in gby.groups.keys()]
    plt.hist(hist_data,
             bins=20,
             histtype='stepfilled', alpha=.5 )
    plt.show()
