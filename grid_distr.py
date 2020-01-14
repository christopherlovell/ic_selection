import os
import numpy as np
import pandas as pd
import gc
from methods import simulation
import matplotlib
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

# delta = np.load('conv_output.npz')['delta']
# delta_log = np.log10(delta)
# delta_sorted = np.sort(delta_log.flatten())
bins = np.linspace(np.min(delta_log)-0.01, np.max(delta_log)+0.01, 26, endpoint = True)

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

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(6, 6), sharex=True, sharey=True,
                            facecolor='w', edgecolor='k')
ax.hist(delta_sorted, density='True', bins=500, label = 'All grids', color = 'black', ls = 'dashed', histtype = 'step', lw = 1.5, stacked = True)
ax.hist(del_grids.flatten(), density='True', bins=bins, label = 'Resim grids', color = 'black', histtype = 'step', lw = 2, stacked = True)

ax.set_xlim(np.min(delta_log)-0.02,np.max(delta_log)+0.02)
ax.grid(True)
ax.set_xlabel(r'log$_{10}$(1+$\delta$)', fontsize = 18)
ax.set_ylabel(r'PDF', fontsize = 18)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(15)
ax.text(0.1, 7.1, '- - All grids', color = 'black', fontsize = 15)
ax.text(0.1, 6.8, '__ ', color = 'black', fontsize = 15)
ax.text(0.14, 6.6, 'Resim grids', color = 'black', fontsize = 15)
fig.tight_layout()
fig.savefig('sampling_bins.pdf')
plt.show()
