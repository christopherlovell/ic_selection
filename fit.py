"""

    Script to load the delta values created using the chosen gridsize and plots
    the gaussian fit to log(1+delta).

"""


import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.spatial.distance import cdist
from methods import simulation, grid_coordinates

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


sim = simulation()
sim.show()

dat = np.load('conv_output.npz')['delta']

delta_log = np.log(dat)
mu, sig = norm.fit(delta_log.flatten())

norm = norm(loc=mu,scale=sig)

#import pickle as pcl
#pcl.dump(norm,open('log_delta_fit.p','wb'))

## Plot distribution and fit
fig, ax = plt.subplots(1,1)
ax.hist(delta_log.flatten(), density='True', bins=500)
x = np.linspace(-2,2,500)
ax.plot(x,norm.pdf(x), lw=1)
ax.set_xlim(np.min(delta_log)-0.1,np.max(delta_log)+0.1)
ax.grid(True)
ax.text(0.2, 2.5, r'$\mu={}$'.format(np.round(mu,4)), fontsize = 14)
ax.text(0.2, 2., r'$\sigma={}$'.format(np.round(sig,4)), fontsize = 14)
ax.set_xlabel(r'ln(1+$\delta$)', fontsize = 15)
ax.set_ylabel(r'PDF', fontsize = 15)
plt.savefig('log_fit.pdf')
plt.show()
