import numpy as np
import matplotlib.pyplot as plt
import massdistribution as md
from numpy.random import normal, power


#mass_samples = normal(20, 1, 1000)

mass_samples = np.concatenate((normal(10, 2, 500), normal(30, 3, 500), normal(25, 6,2000)))
#mass_samples = 3+power(0.4, 5000)*(47)

massfunction = md.DPGMM_mass(mass_samples)
massfunction.compute_dpgmm()
massfunction.display_density()

