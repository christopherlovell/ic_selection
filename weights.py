import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.spatial.distance import cdist
from methods import simulation, grid_coordinates

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle as pcl

sim = simulation()
sim.show()

# ## If already calculated, load...
# norm = pcl.load(open('log_delta_fit.p','rb'))
# print_df = pd.load_csv('GEAGLE_regions_sigma.txt')


## Else...

## Load convolution grid
# dat = np.loadtxt('conv_out.txt')
# dat = dat.reshape((sim.grid,sim.grid,sim.grid))
dat = np.load('conv_output.npz')['delta']

delta_log = np.log(dat)
mu, sig = norm.fit(delta_log.flatten())
norm = norm(loc=mu,scale=sig)


## example for previous selection
regions = np.loadtxt('GEAGLE_regions.txt')
regions[:,1:] /= sim.h
gcood = grid_coordinates(regions[:,1:], sim)

print_df = pd.DataFrame({'x':regions[:,1],'y':regions[:,2],'z':regions[:,3]})
print_df['log(1+delta)'] = [delta_log[tuple(gc)] for gc in gcood]
print_df['delta'] = np.exp(print_df['log(1+delta)']) - 1
print_df['sigma'] = [(delta_log[tuple(gc)] - mu) / sig for gc in gcood]

print("Already selected regions:")
print(print_df)
print_df.to_csv('GEAGLE_regions_sigma.txt')


print_df = print_df.drop([4,6,7,8,9,10,11,12,13])

## Calculate weights including *all* regions

# sort by log(delta+1)
print_df.sort_values(by='log(1+delta)', ascending=True, inplace=True)
print_df = print_df.reset_index()

gb = print_df.groupby('log(1+delta)')
grouped_delta = list(gb.indices.keys())
grouped_index = list(gb.indices.values())

# initialise weights column
print_df['weights'] = 0.

for i,og_idx in enumerate(grouped_index):
    print(og_idx)
    if i==0:
        print_df.loc[og_idx,'lo_lim'] = -.8
        print_df.loc[og_idx,'hi_lim'] = (grouped_delta[i] + grouped_delta[i+1])/2

        weight = (norm.cdf(grouped_delta[i]) +\
                  norm.cdf(grouped_delta[i+1]) ) / 2

        og_idx = print_df.index[grouped_index[i]]
        print_df.loc[og_idx,'weights'] = weight / len(grouped_index[i])
    elif i == len(grouped_delta) - 1:
        print_df.loc[og_idx,'lo_lim'] = (grouped_delta[i-1]+grouped_delta[i])/2
        print_df.loc[og_idx,'hi_lim'] = 1 

        weight = 1 - (norm.cdf(grouped_delta[i-1]) +\
                  norm.cdf(grouped_delta[i]) ) / 2

        og_idx = print_df.index[grouped_index[i]]
        print_df.loc[og_idx,'weights'] = weight / len(grouped_index[i])
    else:
        print_df.loc[og_idx,'lo_lim'] = (grouped_delta[i-1]+grouped_delta[i])/2
        print_df.loc[og_idx,'hi_lim'] = (grouped_delta[i]+grouped_delta[i+1])/2

        weight = (norm.cdf(grouped_delta[i+1]) -\
                  norm.cdf(grouped_delta[i-1]) ) / 2

        og_idx = print_df.index[grouped_index[i]]
        print_df.loc[og_idx,'weights'] = weight / len(grouped_index[i])


print(sum(print_df['weights']))

print_df.to_csv('weights.txt')


## Plot CDF with regions marked ##
gb = print_df.groupby('log(1+delta)')
lo_lims = [temp[1]['lo_lim'].mean() for temp in gb]
hi_lims = [temp[1]['hi_lim'].mean() for temp in gb]

fig,ax = plt.subplots(1,1)

xlim0,xlim1 = -.8,1
ax.set_xlim(xlim0,xlim1)
ax.set_ylim(-.03,1.03)

ax.set_xlabel('$\mathrm{log}(1+\delta)$')
ax.set_ylabel('CDF')

for i,temp in enumerate(gb):
    x = np.linspace(lo_lims[i],hi_lims[i])
    y = norm.cdf(x)
    ax.fill_betweenx(y,xlim0,x,facecolor='C0',edgecolor="white",alpha=0.5)

ax.scatter(print_df['log(1+delta)'],norm.cdf(print_df['log(1+delta)']),s=15,color='black')
x = np.linspace(-.8,1)
ax.plot(x,norm.cdf(x),color='black',linestyle='dashed')

fig.savefig('CDF.png')


