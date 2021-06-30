cimport numpy as np
import numpy as np
from libc.math cimport log, sqrt, M_PI, exp, HUGE_VAL
cimport cython

cdef double LOGSQRT2 = log(sqrt(2*M_PI))

cdef inline double log_add(double x, double y) nogil: return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _log_norm(double x, double x0, double sigma) nogil:
    return -((x-x0)**2)/(2*sigma*sigma) - LOGSQRT2 - log(sigma)

def log_norm(double x, double x0, double sigma):
    return _log_norm(x, x0, sigma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double _log_prob_component(double mu, double mean, double sigma, double w) nogil:
    return log(w) + _log_norm(mu, mean, sigma)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _log_prob_mixture(double mu, double sigma, dict ev):
    cdef double logP = -HUGE_VAL
    cdef dict component
    for component in ev.values():
        logP = log_add(logP,_log_prob_component(mu, component['mean'], sigma, component['weight']))
    return logP

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _integrand(double mu, double sigma, list events, double logN_cnst):
    cdef double logprob = 0.0
    cdef dict ev
    for ev in events:
        logprob += _log_prob_mixture(mu, sigma, ev)
    return exp(logprob - logN_cnst)

def integrand(double sigma,double mu, list events, double m_min, double m_max, double sigma_min, double sigma_max, double logN_cnst):
    return _integrand(mu, sigma, events, logN_cnst)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _compute_norm_const(double mu, double sigma, list events):
    cdef double logprob = 0.0
    cdef dict ev
    for ev in events:
        logprob += _log_prob_mixture(mu, sigma, ev)
    return logprob

def compute_norm_const(double mu, double sigma, list events):
    return _compute_norm_const(mu, sigma, events)
