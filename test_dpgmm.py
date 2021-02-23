import numpy as np
import os
import collapsed_gibbs as DPGMM

events_path = '/home/srinaldi/srinaldi-work1/O3a/Events/'
event_files = [f for f in os.listdir(events_path) if not f.startswith('.')]
events      = []
output      = '/home/srinaldi/srinaldi-work1/O3a/'

for event in event_files:
    events.append(np.genfromtxt(events_path+event))

def normal_density(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def mass_function(m, alpha, m_max, m_min, norm,scale_max=5, scale_min=5):
    return m**(-alpha)*(-alpha+1)/(m_max**(-alpha+1) - m_min**(-alpha+1))*(1-np.exp(-(m-m_max)/scale_max))*(1-np.exp(-(m_min-m)/scale_min))/norm

def norm_mf(pars):
    m = np.linspace(pars[0], pars[1], 1000)
    dm = m[1]-m[0]
    pars.append(1)
    norm = 0.
    for mi in m:
        norm += mass_function(mi, *pars)*dm
    return norm

pars = [30,3]
pars_1 = [30, 3]
pars_2 = [35, 2]


#pars_mf = [1.1,10,60]
#norm = norm_mf(pars_mf)
#pars_mf.append(norm)

sampler = DPGMM.CGSampler(events = events,
                        #mass_b  = [5,50],
                        m_min = 3,
                        m_max = 500,
                        samp_settings = [10000,1000,100],
                        samp_settings_ev = [100,10,100],
                        alpha0  = 1,
                        gamma0   = 1,
                        delta_M = 2,
                        output_folder = output,
                        process_events = False,
                        initial_cluster_number = 3.,
                        n_parallel_threads = 60
                        #injected_density = lambda x : normal_density(x, *pars)
                        # injected_density = lambda x : mass_function(x, *pars_mf)
                        )
                        
sampler.run()
