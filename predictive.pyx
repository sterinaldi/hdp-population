import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport INFINITY
cimport cython
from scipy.integrate import dblquad
from scipy.special import logsumexp


def log_numerical_predictive(events, m_min, m_max, sigma_min, sigma_max):
    events_array = np.empty(len(events), dtype = object)
    for i in range(len(events)):
        ev = events[i]
        components_array = np.empty(len(ev), dtype = object)
        for j in range(len(ev)):
            components_array[j] = np.array([ev[j]['mean'], ev[j]['sigma'], ev[j]['weight']])
        events_array[i] = components_array
    return _log_numerical_predictive(events_array, m_min, m_max, sigma_min, sigma_max)


cdef _log_numerical_predictive(np.ndarray events, double m_min, double m_max, double sigma_min, double sigma_max):
    cdef double I, dI
    integrand = lambda sigma, mu : integrand_function(mu, sigma, events)
    I, dI = dblquad(integrand, m_min, m_max, gfun = lambda x: sigma_min, hfun = lambda x: sigma_max)
    return np.log(I)

cdef double integrand_function(double mu, double sigma, np.ndarray events):
    cdef int i, j
    cdef double x = 0.
    cdef double y
    for i in range(len(events)):
        y = -INFINITY
        for j in range(len(events[i])):
            y = logsumexp([y, np.log(events[i][j][2]) + log_norm(mu, events[i][j][0], sigma, events[i][j][1])])
        x += y
    return np.exp(x)

cdef inline double log_norm(double x, double x0, double sigma1, double sigma2):
    return -((x-x0)**2)/(2*(sigma1**2 + sigma2**2)) - np.log(np.sqrt(2*np.pi)) - 0.5*np.log(sigma1**2 + sigma2**2)
