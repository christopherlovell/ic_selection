
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.spatial.distance import cdist
from methods import simulation, grid_coordinates

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


sim = simulation()
sim.show()

dat = np.loadtxt('conv_out.txt')
dat = dat.reshape((sim.grid,sim.grid,sim.grid))

delta_log = np.log(dat)
mu, sig = norm.fit(delta_log.flatten())

import pickle as pcl
norm = pcl.load(open('log_delta_fit.p','rb'))


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

## Calculate weights including *all* regions

# sort by log(delta+1)
print_df.sort_values(by='log(1+delta)', ascending=True, inplace=True)

gb = print_df.groupby('log(1+delta)')
grouped_delta = list(gb.indices.keys())
grouped_index = list(gb.indices.values())

# initialise weights column
print_df['weights'] = 0.

for i in np.arange(len(grouped_delta)):

    print(i)

    if i==0:
        weight = (norm.cdf(grouped_delta[i]) +\
                  norm.cdf(grouped_delta[i+1]) ) / 2

        print(weight)

        og_idx = print_df.index[grouped_index[i]]
        print_df.loc[og_idx,'weights'] = weight / len(grouped_index[i])
    elif i == len(grouped_delta) - 1:
        weight = 1 - (norm.cdf(grouped_delta[i-1]) +\
                  norm.cdf(grouped_delta[i]) ) / 2

        print(weight)
        og_idx = print_df.index[grouped_index[i]]
        print_df.loc[og_idx,'weights'] = weight / len(grouped_index[i])
    else:
        weight = (norm.cdf(grouped_delta[i+1]) -\
                  norm.cdf(grouped_delta[i-1]) ) / 2

        print(weight)
        og_idx = print_df.index[grouped_index[i]]
        print_df.loc[og_idx,'weights'] = weight / len(grouped_index[i])


print(sum(print_df['weights']))




