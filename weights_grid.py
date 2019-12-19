import os
import numpy as np
import pandas as pd
import gc
from methods import simulation


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

## Read in the resimmed regions
print_df = pd.read_csv('GEAGLE_regions.txt')

pos = (np.array(print_df[['x','y','z']])/sim.conv).astype(int)
weights = np.zeros(len(pos))

bins = np.linspace(np.min(delta_log)-0.01, np.max(delta_log)+0.01, 26, endpoint = True)

hist, edges = np.histogram(delta_log, bins = bins)
ntot = np.sum(hist)

gc.collect()

grid_weights = np.zeros(np.shape(delta))

for ii in range(len(bins)-1):

    ok = np.logical_and(delta_log >= bins[ii], delta_log < bins[ii+1])

    if np.sum(ok)>0:

        grid_weights[ok] = hist[ii]/ntot

gc.collect()

xcoord, ycoord, zcoord = get_xyz(sim)
for jj in range(len(pos)):

    xx = pos[jj][0] + xcoord
    yy = pos[jj][1] + ycoord
    zz = pos[jj][2] + zcoord

    xx[xx>=sim.grid]-=sim.grid
    yy[yy>=sim.grid]-=sim.grid
    zz[zz>=sim.grid]-=sim.grid

    weights[jj] = np.sum(grid_weights[(xx,yy,zz)])

weights*=(1./np.sum(weights))
print_df['weights'] = weights
print_df.to_csv('weights_grid.txt')
print (print_df)
