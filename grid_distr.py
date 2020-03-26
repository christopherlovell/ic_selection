import os
import numpy as np
import pandas as pd
import gc
from methods import simulation
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")

def get_xyz(sim):

    r, boxl, gridsize = sim.r,sim.boxsize,sim.grid

    dl = boxl / gridsize
    sz = r/dl
    length = 2*round(sz)

    ker = np.zeros((length+1,length+1,length+1))

    x, y, z = (length)/2., (length)/2., (length)/2.

    sz_squared = sz**2
    sel = np.arange(length+1)
    xcoord = np.array([])
    ycoord = np.array([])
    zcoord = np.array([])
    for i in sel:
        for j in sel:
            for k in sel:
                if (i-x)**2 + (j-y)**2 + (k-z)**2 <= sz_squared:
                    xcoord = np.append(xcoord, i)
                    ycoord = np.append(ycoord, j)
                    zcoord = np.append(zcoord, k)

    xcoord-=x
    ycoord-=y
    zcoord-=z

    return xcoord.astype(np.int), ycoord.astype(np.int), zcoord.astype(np.int)

sim = simulation()
sim.show()
nbins = 50

delta = np.load('conv_output.npz')['delta']
delta_log = np.log10(delta)
delta_sorted = np.sort(delta_log.flatten())
bins = np.linspace(np.min(delta_log)-0.01, np.max(delta_log)+0.01, nbins+1, endpoint = True)

mu, sig = norm.fit(delta_log.flatten())
norm = norm(loc=mu,scale=sig)

## Read in the resimmed regions
print_df = pd.read_csv('GEAGLE_regions.txt')
print_df = (print_df.sort_values(by=['delta'], ascending=True)).reset_index(drop=True)
pos = (np.array(print_df[['x','y','z']])/sim.conv).astype(int)
nsims = len(pos)
del_centre = np.array(print_df['log(1+delta)'])
xcoord, ycoord, zcoord = get_xyz(sim)
del_grids = np.zeros((nsims, len(xcoord)))

for jj in range(len(pos)):

    xx = pos[jj][0] + xcoord
    yy = pos[jj][1] + ycoord
    zz = pos[jj][2] + zcoord

    xx[xx>=sim.grid]-=sim.grid
    yy[yy>=sim.grid]-=sim.grid
    zz[zz>=sim.grid]-=sim.grid

    del_grids[jj] = delta_log[(xx,yy,zz)]


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(7, 6), sharex=True, sharey=True,
                            facecolor='w', edgecolor='k')
ax1 = ax.twinx()

ax.hist(delta_log.flatten(), density='True', histtype = 'stepfilled', bins=150)
n, nbins = np.histogram(del_grids.flatten(), bins=bins)
x = np.linspace(-1,1,500)
ax.plot(x,norm.pdf(x), lw=1.5, color='black', ls='dashed', label='All grids')
ax.hist(del_grids.flatten(), density='True', bins=bins, label = 'Resim grids', color = 'black', histtype = 'step', lw = 1.5)
ax1.hist(del_grids.flatten(), density='True', bins=bins, label = 'Resim grids', color = 'black', histtype = 'step', lw = 1.5, alpha = 0.)

twinyticks = np.array([100, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3])
twinyticklabels = ['100', '1000', '2000', '3000', '4000', '5000', '6000', '7000']
yticks = twinyticks/(np.sum(n)*(bins[1]-bins[0]))

ax.text(-0.27, 2.5, r'$\mu={}$'.format(np.round(mu,4)), fontsize = 13)
ax.text(-0.27, 2., r'$\sigma={}$'.format(np.round(sig,4)), fontsize = 13)

ax.set_xlim(np.min(delta_log)-0.02,np.max(delta_log)+0.02)
ax.grid(True, alpha = 0.4)
ax.set_ylim(0., 8)
ax1.set_ylim(0., 8)
ax.set_xlabel(r'log$_{10}$(1+$\delta$)', fontsize = 13)
ax.set_ylabel(r'PDF', fontsize = 13)

ax1.set_ylabel(r'N', fontsize = 13)
ax1.set_yticks(yticks)
ax1.set_yticklabels(twinyticklabels, fontsize = 13)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(12)

ax.text(0.1, 7.1, '- - All grids', color = 'black', fontsize = 13)
ax.text(0.1, 6.8, '---', color = 'black', fontsize = 13)
ax.text(0.125, 6.7, 'Resim grids', color = 'black', fontsize = 13)


fig.savefig('sampling_bins.pdf')
plt.show()
