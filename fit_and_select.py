
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.spatial.distance import cdist
from methods import near_coords,simulation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


sim = simulation()
sim.show()

dat = np.loadtxt('conv_out.txt')
dat = dat.reshape((sim.grid,sim.grid,sim.grid))

delta_log = np.log(dat)
mu, sig = norm.fit(delta_log.flatten())



## Plot distribution and fit
fig, ax = plt.subplots(1,1)
plt.hist(delta_log.flatten(), density='normed',bins=100)
x = np.linspace(-3,3,100)
plt.plot(x,norm(loc=mu,scale=sig).pdf(x))
plt.xlim(-2.5,2.5)
plt.savefig('log_fit.png')
# plt.show()


def selection(x,centre,dx,tol=5e-3):
    mask = (x-centre) < (dx+tol)
    mask = mask & ((x-centre) > (dx-tol))
    return np.array(np.where(mask))


# to select regions of a given overdensity, set `dx`
# choose `tol` to adjust how near you select

# for example, to select 2-sigma regions:
sel_2s = selection(delta_log, centre=mu, dx=2*sig, tol=3e-3).T

print('N:',sel_2s.shape)
#print('log(1+delta) :',delta_log[sel_2s[:10]],'...')
#print('1+delta :',delta_log[sel_2s[:10]],'...')

## Coordinates:
coods = sel_2s * sim.conv

## get overlapping regions
def overlapping_regions(coods,delta_log,sim):
    """
    return mask for array of coordinates indicating overlapping regions

    If coordinates is too large, will return a memory error

    Args:
        coods (array,float): 2D array of coordinates
        sim (object)
    """    
    out_mask = np.zeros(len(coods),dtype='bool')
    for c in coods[np.argsort(delta_log)]:  # inefficient, should update mask after each exclusion
        d = cdist([c],coods)[0]
        overlap = (d < sim.r) 
        idx = np.where(overlap)[0][np.argmax(delta_log[overlap])]
        out_mask[idx] = True
    
    return out_mask


dl = np.array([delta_log[tuple(s2)] for s2 in sel_2s])
olap_mask = overlapping_regions(coods[:100],dl[:100],sim=sim)

print_df = pd.DataFrame({'x':coods[olap_mask,0],'y':coods[olap_mask,1],'z':coods[olap_mask,2]})
print_df['log(1+delta)'] = dl[olap_mask]
print_df['delta'] = np.exp(print_df['log(1+delta)']) - 1
print_df['sigma'] = (dl[olap_mask] - mu) / sig
print("2-sigma regions:")
print(print_df)

# print('Non-overlapping regions:',coods[:100][olap_mask])

## Find overdensity / sigma for given coordinates
def grid_coordinates(coods, sim):
    return (coods / sim.conv).astype(int)


# example for previous selection
regions = np.loadtxt('GEAGLE_regions.txt')
regions[:,1:] /= sim.h
gcood = grid_coordinates(regions[:,1:], sim)

import pandas as pd
print_df = pd.DataFrame({'x':regions[:,1],'y':regions[:,2],'z':regions[:,3]})
print_df['log(1+delta)'] = [delta_log[tuple(gc)] for gc in gcood]
print_df['delta'] = np.exp(print_df['log(1+delta)']) - 1 
print_df['sigma'] = [(delta_log[tuple(gc)] - mu) / sig for gc in gcood]

print("Already selected regions:")
print(print_df)


## Find most overdense regions
# don't need to sort the whole array, just most overdense regions
upper_mask = np.asarray(np.where(delta_log > (mu + 5.2)*sig)).T
sorted_idx = np.argsort([delta_log[tuple(um)] for um in upper_mask])[::-1]
coods = upper_mask * sim.conv

coods = upper_mask * sim.conv
dl = np.array([delta_log[tuple(um)] for um in upper_mask])
olap_mask = overlapping_regions(coods,dl,sim)

print_df = pd.DataFrame({'x':coods[olap_mask,0],'y':coods[olap_mask,1],'z':coods[olap_mask,2]})
print_df['log(1+delta)'] = dl[olap_mask]
print_df['delta'] = np.exp(print_df['log(1+delta)']) - 1 
print_df['sigma'] = (dl[olap_mask] - mu) / sig

print("Newly selected most overdense regions:")
print(print_df)

