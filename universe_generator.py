import numpy as np
from numpy.random import uniform, triangular, normal, gumbel
from scipy.special import erf
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt

#def mass_function(m, alpha, m_max, m_min,scale_max=5, scale_min=5):
#    return m**(-alpha)*(-alpha+1)/(m_max**(-alpha+1) - m_min**(-alpha+1))*(1-np.exp(-(m-m_max)/scale_max))*(1-np.exp(-(m_min-m)/scale_min))
def mass_function(m, alpha=0, m_max=50, m_min= 20, l_max = 5, l_min = 5):
    return m**(-alpha)*(1+erf((m-m_min)/(l_min)))*(1+erf((m_max-m)/l_max))/4.

#def mass_function(m, x01 = 40, sigma1 = 6, x02 = 60, sigma2 = 4.5):
#    return (0.8*np.exp(-(m-x01)**2/(2*sigma1**2))/(np.sqrt(2*np.pi)*sigma1) + 0.2*np.exp(-(m-x02)**2/(2*sigma2**2))/(np.sqrt(2*np.pi)*sigma2))/2.


def posterior_probability(m, m_true, k, b):
    norm = 2*b*np.sqrt(b/k)
    return (-k*(m-m_true)**2+b)/norm

def mass_sampler(m_max, m_min, sup):
    while 1:
        mass_try = uniform(m_min, m_max)
        if mass_function(mass_try) > uniform(0, sup):
            return mass_try

def sigma_sampler(s_min, s_max):
    return np.exp(uniform(np.log(s_min), np.log(s_max)))
def posterior_sampler(m_true, sigma):
    #return triangular(m_true*0.8, m_true, m_true*1.1)
    return normal(m_true, sigma)
    
if __name__ == '__main__':

    out_folder = '/Users/stefanorinaldi/Documents/mass_inference/selfunc/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    post_folder = out_folder+'/events/'
    plot_folder = out_folder+'/plots/'

    if not os.path.exists(out_folder+'/events/'):
        os.mkdir(post_folder)
        
    data_sf = np.genfromtxt(out_folder + 'selfunc.txt', names = True)
    selfunc = interp1d(data_sf['m1'], data_sf['pdet']**3*(np.max( data_sf['pdet'])/np.max(data_sf['pdet']**3)))
    
    n_bbh = 250
    n_samples = 100

    alpha = 1.1
    m_min = 16
    m_max = 70
    s_min = 2
    s_max = 4
    
    app = np.linspace(m_min, m_max, 1000)
    mf  = mass_function(app)
    sup = mf.max()
    norm = np.sum(mf*selfunc(app)*(app[1]-app[0]))

    bbhs = []
    masses = []
    sigmas = []
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    i = 0
#    for i in range(n_bbh):
    while i < n_bbh:
        m1 = mass_sampler(m_max, m_min, sup)
        if selfunc(m1) > uniform():
            sigma = np.exp(uniform(np.log(3), np.log(5)))
            samples = normal(loc = m1, scale = sigma, size = n_samples)
            np.savetxt(post_folder + '/event_{0}.txt'.format(i+1), np.array(samples))
            bbhs.append(samples)
            masses.append(m1)
            sigmas.append(sigma)
            i += 1


    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    appM = np.linspace(m_min, m_max, 1000)
    ax1.hist(masses, bins = int(np.sqrt(len(masses))), density = True)
    ax1.plot(appM, mass_function(appM)*selfunc(appM)/norm, color = 'r', label = 'Observed')
    ax1.plot(appM, mass_function(appM)/(np.sum(mass_function(appM))*(appM[1]-appM[0])), color = 'k', label = 'Astrophysical')
#    ax1.plot(appM, selfunc(appM))
    ax1.set_xlabel('$M_1\ [M_\\odot]$')
    ax1.legend(loc = 0)
#    ax1.set_ylim(0, sup*(1.1))
    plt.tight_layout()
    plt.savefig(out_folder+'/truths.pdf', bbox_inches = 'tight')
    np.savetxt(out_folder+'/truths.txt', np.array([masses, sigmas]).T, header = 'm sigma')

    flattened_m = np.array([m for ev in bbhs for m in ev])

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.hist(flattened_m, bins = 1000, density = True)
    ax.plot(appM, mass_function(appM)/norm, color = 'r', linewidth = 0.5)
    ax.set_xlabel('$M_1\ [M_\\odot]$')
    fig.savefig(out_folder+'/all_samples.pdf', bbox_inches = 'tight')
