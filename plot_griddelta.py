import os
import numpy as np
import pandas as pd
import gc
from methods import simulation
import matplotlib
import matplotlib.pyplot as plt

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
delta_log = np.log(delta)
delta_sorted = np.sort(delta_log.flatten())
num = np.arange(1, len(delta_sorted)+1, 1)/(len(delta_sorted)+1)


norm = matplotlib.colors.Normalize(vmin=delta_sorted[0]-0.01, vmax=delta_sorted[-1]+0.01)
# choose a colormap
c_m = matplotlib.cm.viridis_r
# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])

## Read in the resimmed regions
print_df = pd.read_csv('GEAGLE_regions.txt',delim_whitespace=True)
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

sims = np.linspace(0, 1, nsims)

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8), sharex=True, sharey=True,
                            facecolor='w', edgecolor='k')


axs.plot(delta_sorted, num, color='crimson')

for jj in range(nsims):

    axs.scatter(del_grids[jj], np.ones(len(xcoord))*sims[jj], s = 2, color=s_m.to_rgba(del_centre[jj]))

#axs.axhline(y = 0, lw = 1)

cbaxes = fig.add_axes([0.2, 0.5, 0.03, 0.35])
fig.colorbar(s_m, cax=cbaxes)
cbaxes.set_ylabel(r'ln(1+$\delta_{\mathrm{centre}}$)', fontsize = 18)
axs.set_xlabel(r'ln(1+$\delta$)', fontsize = 18)
axs.set_ylabel(r'CDF', fontsize = 18)
plt.savefig('cum_delta.png')
plt.close()
