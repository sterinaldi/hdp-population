import numpy as np
import os
import collapsed_gibbs as DPGMM
import cpnest.model
import matplotlib.pyplot as plt

def sort_matrix(a, axis = -1):
    keys = np.copy(a[:,axis])
    sorted_keys = np.sort(np.copy(keys))
    indexes = [np.where(el == sorted_keys)[0][0] for el in keys]
    sorted_a = np.array([np.copy(a[i]) for i in indexes])
    return sorted_a
    

class point:
    def __init__(self, fz, dfz, DA, dDA):
        self.fz = fz
        self.dfz = dfz
        self.DA = DA
        self.dDA = dDA

class Fit(cpnest.model.Model):

    def __init__(self, data, *args, **kwargs):

        super(Fit,self).__init__()
        self.data  = data
        self.names  = ['H0']
        self.bounds = [[10,100]]

    def log_prior(self, x):
        logP = super(Fit,self).log_prior(x)
        if np.isfinite(logP):
            return 0.
        else:
            return -np.inf

    def log_likelihood(self, x):
        logL = 0.
        for point in self.data:
            fz = point.fz
            dfz = point.dfz
            DA = point.DA
            dDA = point.dDA
            
            logL += ((fz - x['H0']*DA)**2/(2*((x['H0']*dDA)**2. + dfz**2))) - 0.5*np.log(((x['H0']*dDA)**2. + dfz**2))

        return logL

catalog_file = '/Users/stefanorinaldi/Documents/mass_inference/galcluster/catalog.txt'
catalog = [sort_matrix(np.genfromtxt(catalog_file), axis = 2)]
output = '/Users/stefanorinaldi/Documents/mass_inference/galcluster/'

if not os.path.exists(output):
    os.mkdir(output)

density = 0.66
process_data = True


if process_data:
    sampler = DPGMM.CGSampler(events = catalog,
                            n_draws = 10,
                            burnin  = 100,
                            step    = 10,
                            alpha0  = 1,
                            output_folder = output)
    sampler.run_event_sampling()

clusters_folder = output+'/clusters/'
cluster_files = [clusters_folder + f for f in os.listdir(clusters_folder) if not f.startswith('.')]

clusters = []
names = ['ra', 'dec', 'z', 'rr','rd','rz', 'dr', 'dd', 'dz', 'zr','zd','zz', 'N']

for file in cluster_files:
    clusters.append(np.genfromtxt(file, names = names))

z = []
dz = []
DA = []
dDA = []


for cl in clusters:
    z.append(cl['z'].mean()/10**2)
    dz.append(cl['z'].std()/10**2)
    N = cl['N'].mean()
    dN = cl['N'].std()
    print((np.sqrt(cl['rr']).mean()))
    theta = np.arcsin(3*np.sin((np.sqrt(cl['rr']).mean())))
    dtheta = 3*np.cos(np.sqrt(cl['rr']).mean())*(np.sqrt(cl['rr']).std())/np.sqrt(1-(3*np.sin(np.sqrt(cl['rr']).mean()))**2)
    ri = (3*N/(4*np.pi*density))**(1./3.)
    dri = (6./(12*np.pi*density))*dN/(ri**2)
    angdist = ri/theta
    dangdist = angdist*np.sqrt((dri/ri)**2 + (dtheta/theta)**2)
    DA.append(angdist)
    dDA.append(dangdist)

z = np.array(z)
dz = np.array(dz)
DA = np.array(DA)
dDA = np.array(dDA)
fz = z*(1+z)**2
dfz = ((1+z)**2 + 2*z*(1+z))*dz

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.errorbar(DA, fz, xerr = dDA, yerr = dfz, marker = '+', ls = '')
ax.set_xlabel('$D_A$ [Mpc]')
ax.set_ylabel('$z(1+z)^2$')
plt.savefig(output+'datapoint.pdf', bbox_inches = 'tight')

data = []
for fzi, dfzi,DAi, dDAi in zip(fz, dfz, DA, dDA):
    data.append(point(fzi, dfzi, DAi/(3*10**5), dDAi/(3*10**5))) # c

H = Fit(data)
work = cpnest.CPNest(H, verbose = 0, output = output)
#work.run()

samples_H0 = work.get_posterior_samples()['H0']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('$H_0$')
ax.hist(samples_H0, bins = int(np.sqrt(len(samples_H0))))
plt.savefig(output+'H0.pdf', bbox_inches = 'tight')
