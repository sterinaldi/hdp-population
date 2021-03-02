import numpy as np
from numpy.random import uniform, triangular, normal, gumbel
from scipy.special import erf
import os
import matplotlib.pyplot as plt

#def mass_function(m, alpha, m_max, m_min,scale_max=5, scale_min=5):
#    return m**(-alpha)*(-alpha+1)/(m_max**(-alpha+1) - m_min**(-alpha+1))*(1-np.exp(-(m-m_max)/scale_max))*(1-np.exp(-(m_min-m)/scale_min))
def mass_function(m, alpha=1.1, m_max=70, m_min=15, l_max = 5, l_min = 5):
    return m**(-alpha)*(1+erf((m-m_min)/(l_min)))*(1+erf((m_max-m)/l_max))/4.

#def mass_function(m, x01 = 30, sigma1 = 6., x02 = 80, sigma2 = 10):
#    return (np.exp(-(m-x01)**2/(2*sigma1**2))/(np.sqrt(2*np.pi)*sigma1) + np.exp(-(m-x02)**2/(2*sigma2**2))/(np.sqrt(2*np.pi)*sigma2))/2.


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

    out_folder = '/Users/stefanorinaldi/Documents/mass_inference/PL/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    post_folder = out_folder+'/events/'
    plot_folder = out_folder+'/plots'

    if not os.path.exists(out_folder+'/events/'):
        os.mkdir(post_folder)
        
        
    n_bbh = 1000
    n_samples = 100

    alpha = 1.1
    m_min = 3
    m_max = 100
    s_min = 2
    s_max = 4
    
    app = np.linspace(m_min, m_max, 1000)
    mf  = mass_function(app)
    sup = mf.max()
    norm = np.sum(mf*(app[1]-app[0]))

    bbhs = []
    masses = []
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for i in range(n_bbh):
        samples = []
        m1 = mass_sampler(m_max, m_min, sup)
#        m1 = normal(30, 6)
        sigma1 = sigma_sampler(s_min, s_max)
        #samples = normal(m,sigma,n_samples)
#        for _ in range(n_samples):
            #samples.append(posterior_sampler(m, sigma))#,k,b))
            #samples.append(mass_sampler(alpha, m_max, m_min))
        samples1 = gumbel(m1, sigma1, n_samples)
#        ax.clear()
#        app = np.linspace(max(samples), min(samples), 1000)
        np.savetxt(post_folder+'/event_{0}_1.txt'.format(i+1), np.array(samples1))
#        np.savetxt(post_folder+'/event_{0}_2.txt'.format(i+1), np.array(samples2))
#        fig.suptitle('Event {0}'.format(i+1))
#        ax.hist(samples, bins = int(np.sqrt(len(samples))), density = True)
#        ax.set_xlabel('$M_1\ [M_\\odot]$')
#        plt.savefig(plot_folder+'/event_{0}.pdf'.format(i+1), bbox_inches = 'tight')
        bbhs.append(samples)
        masses.append(m1)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    appM = np.linspace(m_min, m_max, 1000)
    ax1.hist(masses, bins = int(np.sqrt(len(masses))), density = True)
    ax1.plot(appM, mass_function(appM)/norm, color = 'r', linewidth = 0.5)
    ax1.set_xlabel('$M_1\ [M_\\odot]$')
    plt.tight_layout()
    plt.savefig(out_folder+'/truths.pdf', bbox_inches = 'tight')
    np.savetxt(out_folder+'/truths.txt', np.array(masses))
#
#    flattened_m = np.array([m for ev in bbhs for m in ev])
#
#    fig = plt.figure()
#    ax  = fig.add_subplot(111)
#    ax.hist(flattened_m, bins = 1000, density = True)
#    ax.plot(appM, mass_function(appM, alpha, m_max, m_min)/norm, color = 'r', linewidth = 0.5)
#    ax.set_xlabel('$M_1\ [M_\\odot]$')
#    fig.savefig(out_folder+'/all_samples.pdf', bbox_inches = 'tight')
