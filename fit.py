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

delta_log = np.log10(dat)
mu, sig = norm.fit(delta_log.flatten())

norm = norm(loc=mu,scale=sig)

## Plot distribution and fit
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8), sharex=True, sharey=True,
                            facecolor='w', edgecolor='k')
ax.hist(delta_log.flatten(), density='True', bins=200)
x = np.linspace(-1,1,500)
ax.plot(x,norm.pdf(x), lw=1, color = 'black')
ax.set_xlim(np.min(delta_log)-0.02,np.max(delta_log)+0.02)
ax.set_ylim(0, 7.7)
ax.grid(True, alpha = 0.5)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(15)
ax.text(0.15, 2.5, r'$\mu={}$'.format(np.round(mu,4)), fontsize = 15)
ax.text(0.15, 2., r'$\sigma={}$'.format(np.round(sig,4)), fontsize = 15)
ax.set_xlabel(r'log$_{10}$(1+$\delta$)', fontsize = 18)
ax.set_ylabel(r'PDF', fontsize = 18)
fig.tight_layout()
plt.savefig('log_fit.pdf')
plt.show()
