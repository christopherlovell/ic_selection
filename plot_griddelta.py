import os
import numpy as np
import pandas as pd
import gc
from methods import simulation
import matplotlib
matplotlib.rcParams['text.usetex'] = True
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

delta = np.load('conv_output.npz')['delta']
delta_log = np.log10(delta)
delta_sorted = np.sort(delta_log.flatten())


norm = matplotlib.colors.Normalize(vmin=delta_sorted[0]-0.01, vmax=delta_sorted[-1]+0.01)
# choose a colormap
c_m = matplotlib.cm.viridis_r
# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])

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

sims = np.linspace(0.15, 7.55, nsims)

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8), sharex=True, sharey=True,
                            facecolor='w', edgecolor='k')


for jj in range(nsims):

    axs.scatter(del_grids[jj], np.ones(len(xcoord))*sims[jj], s = 2, color=s_m.to_rgba(del_centre[jj]))
    axs.scatter(del_centre[jj], sims[jj], s = 25, color='black', marker='x')

#axs.axhline(y = 0, lw = 1)
axs.set_xlim(np.min(delta_log)-0.02,np.max(delta_log)+0.02)
axs.set_ylim(0, 7.7)
cbaxes = fig.add_axes([0.2, 0.5, 0.03, 0.35])
fig.colorbar(s_m, cax=cbaxes)
cbaxes.set_ylabel(r'log$_{10}$(1+$\delta_{\mathrm{centre}}$)', fontsize = 18)
axs.set_xlabel(r'log$_{10}$(1+$\delta$)', fontsize = 18)
axs.set_ylabel(r'PDF', fontsize = 18)
axs.yaxis.label.set_color('white')
axs.tick_params(axis='y', colors='white')
axs.xaxis.grid(True, alpha=0.4)
for label in (axs.get_xticklabels() + cbaxes.get_yticklabels()):
    label.set_fontsize(15)

plt.savefig('cum_delta.pdf')
plt.close()
