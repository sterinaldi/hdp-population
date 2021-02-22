import numpy as np
from numpy.random import uniform, triangular, normal
import os
import matplotlib.pyplot as plt

def mass_function(m, alpha, m_max, m_min,scale_max=5, scale_min=5):
    return m**(-alpha)*(-alpha+1)/(m_max**(-alpha+1) - m_min**(-alpha+1))*(1-np.exp(-(m-m_max)/scale_max))*(1-np.exp(-(m_min-m)/scale_min))

def posterior_probability(m, m_true, k, b):
    norm = 2*b*np.sqrt(b/k)
    return (-k*(m-m_true)**2+b)/norm

def mass_sampler(alpha, m_max, m_min):
    while 1:
        mass_try = uniform(m_min, m_max)
        if mass_function(mass_try, alpha, m_max, m_min) > uniform(0, mass_function(m_min+(1.2*5), alpha, m_max, m_min)):
            return mass_try

def sigma_sampler(s_min, s_max):
    return np.exp(uniform(np.log(s_min), np.log(s_max)))
def posterior_sampler(m_true, sigma):
    #return triangular(m_true*0.8, m_true, m_true*1.1)
    return normal(m_true, sigma)
    
if __name__ == '__main__':

    out_folder = '/Users/stefanorinaldi/Documents/mass_inference/universe_2/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    post_folder = out_folder+'/events/'
    plot_folder = out_folder+'/plots'

    if not os.path.exists(out_folder+'/events/'):
        os.mkdir(post_folder)
    if not os.path.exists(out_folder+'/plots/'):
        os.mkdir(plot_folder)

    n_bbh = 200
    n_samples = 20

    alpha = 1.1
    m_min = 10
    m_max = 60
    s_min = 2
    s_max = 5

    bbhs = []
    masses = []
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for i in range(n_bbh):
        samples = []
        m = mass_sampler(alpha, m_max, m_min)
        # m = normal(30, 3)
        sigma = sigma_sampler(s_min, s_max)
#        for _ in range(n_samples):
            #samples.append(posterior_sampler(m, sigma))#,k,b))
            #samples.append(mass_sampler(alpha, m_max, m_min))
        samples = normal(m, sigma, n_samples)
        ax.clear()
        app = np.linspace(max(samples), min(samples), 1000)
        np.savetxt(post_folder+'/event_{0}.txt'.format(i+1), np.array(samples))
        fig.suptitle('Event {0}'.format(i+1))
        ax.hist(samples, bins = int(np.sqrt(len(samples))), density = True)
        ax.set_xlabel('$M_1\ [M_\\odot]$')
        plt.savefig(plot_folder+'/event_{0}.pdf'.format(i+1), bbox_inches = 'tight')
        bbhs.append(samples)
        masses.append(m)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    appM = np.linspace(m_min, m_max, 1000)
    ax1.hist(masses, bins = int(np.sqrt(len(masses))), density = True)
    ax1.plot(appM, mass_function(appM, alpha, m_max, m_min), color = 'r', linewidth = 0.5)
    ax1.set_xlabel('$M_1\ [M_\\odot]$')
    plt.tight_layout()
    plt.savefig(out_folder+'/truths.pdf', bbox_inches = 'tight')

    flattened_m = np.array([m for ev in bbhs for m in ev])

    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.hist(flattened_m, bins = 1000, density = True)
    ax.plot(appM, mass_function(appM, alpha, m_max, m_min), color = 'r', linewidth = 0.5)
    ax.set_xlabel('$M_1\ [M_\\odot]$')
    fig.savefig(out_folder+'/all_samples.pdf', bbox_inches = 'tight')
