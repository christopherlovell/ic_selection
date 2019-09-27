import numpy as np
import pandas as pd

from scipy.stats import norm
from methods import simulation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
print_df = pd.read_csv('GEAGLE_regions.txt',delim_whitespace=True)
# regions[['x','y','z']] /= sim.h

print("Already selected regions:")
print(print_df)

#  ## example for previous selection
#  regions = pd.read_csv('GEAGLE_regions.txt',delim_whitespace=True)
#  # regions[['x','y','z']] /= sim.h
#  gcood = grid_coordinates(regions[['x','y','z']], sim)
#
#  print_df = pd.DataFrame({'x':regions['x'],'y':regions['y'],'z':regions['z']})
#  print_df['log(1+delta)'] = delta_log[gcood['x'],gcood['y'],gcood['z']] #[delta_log[tuple(gc)] for gc in gcood]
#  print_df['delta'] = np.exp(print_df['log(1+delta)']) - 1
#  print_df['sigma'] = (delta_log[gcood['x'],gcood['y'],gcood['z']] - mu) / sig # [(delta_log[tuple(gc)] - mu) / sig for gc in gcood]
#
#  print("Already selected regions:")
#  print(print_df)
#  print_df.to_csv('GEAGLE_regions_sigma.txt')


## Drop some regions to ensure correct weights
al = [15,31,33] 
lo1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,31,33,35,37]
lo = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]
print_df = print_df.drop(al)


# sort by log(delta+1)
# print_df.sort_values(by='log(1+delta)', ascending=True, inplace=True)
# print_df = print_df.reset_index()

# bin by log(1+delta) (choose some arbitrary binning)
bins = np.arange(-0.4, 0.75, 0.05)
binning = pd.cut(print_df['log(1+delta)'],bins)
gb = print_df.groupby(binning)

# find mean overdensity
mean_od = gb.mean()['log(1+delta)'].dropna()
lo_lim = np.hstack([np.min(delta_log)-0.01,(mean_od.values[1:] + mean_od.values[:-1]) / 2])
hi_lim = np.hstack([(mean_od.values[1:] + mean_od.values[:-1]) / 2, np.max(delta_log)+0.01])

weights = norm.cdf(hi_lim) - norm.cdf(lo_lim)

# assign weights to dataframe
# divide by number of regions
jj = 0
for i,(key,item) in enumerate(gb):
    if (item.empty == False):
        print_df.loc[item.index,'weights'] = weights[jj] / len(item) 
        jj+=1
    

print(sum(print_df['weights']))

# gb = print_df.groupby('log(1+delta)')
# grouped_delta = list(gb.indices.keys())
# grouped_index = list(gb.indices.values())
# 
# # initialise weights column
# print_df['weights'] = 0.
# 
# for i,og_idx in enumerate(grouped_index):
#     print(og_idx)
#     if i==0:
#         print_df.loc[og_idx,'lo_lim'] = -.8
#         print_df.loc[og_idx,'hi_lim'] = (grouped_delta[i] + grouped_delta[i+1])/2
# 
#         weight = (norm.cdf(grouped_delta[i]) +\
#                   norm.cdf(grouped_delta[i+1]) ) / 2
# 
#         og_idx = print_df.index[grouped_index[i]]
#         print_df.loc[og_idx,'weights'] = weight / len(grouped_index[i])
#     elif i == len(grouped_delta) - 1:
#         print_df.loc[og_idx,'lo_lim'] = (grouped_delta[i-1]+grouped_delta[i])/2
#         print_df.loc[og_idx,'hi_lim'] = 1 
# 
#         weight = 1 - (norm.cdf(grouped_delta[i-1]) +\
#                   norm.cdf(grouped_delta[i]) ) / 2
# 
#         og_idx = print_df.index[grouped_index[i]]
#         print_df.loc[og_idx,'weights'] = weight / len(grouped_index[i])
#     else:
#         print_df.loc[og_idx,'lo_lim'] = (grouped_delta[i-1]+grouped_delta[i])/2
#         print_df.loc[og_idx,'hi_lim'] = (grouped_delta[i]+grouped_delta[i+1])/2
# 
#         weight = (norm.cdf(grouped_delta[i+1]) -\
#                   norm.cdf(grouped_delta[i-1]) ) / 2
# 
#         og_idx = print_df.index[grouped_index[i]]
#         print_df.loc[og_idx,'weights'] = weight / len(grouped_index[i])
# 
# 
# print(sum(print_df['weights']))

print_df.to_csv('weights_cdf.txt')

## Plot CDF with regions marked ##
# gb = print_df.groupby('log(1+delta)')
# lo_lims = [temp[1]['lo_lim'].mean() for temp in gb]
# hi_lims = [temp[1]['hi_lim'].mean() for temp in gb]

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(5,10))

xlim0,xlim1 = -.8,1
ax1.set_xlim(xlim0,xlim1)
ax1.set_ylim(-.03,1.03)

ax1.set_xlabel('$\mathrm{log}(1+\delta)$')
ax1.set_ylabel('$\mathrm{CDF}$')

for i,temp in enumerate(mean_od):
    x = np.linspace(lo_lim[i],hi_lim[i])
    y = norm.cdf(x)
    ax1.fill_betweenx(y,xlim0,x,facecolor='C0',edgecolor="white",alpha=0.5)

ax1.scatter(print_df['log(1+delta)'],norm.cdf(print_df['log(1+delta)']),s=15,color='black')
x = np.linspace(-.8,1)
ax1.plot(x,norm.cdf(x),color='black',linestyle='dashed')

# fig.savefig('CDF.png')

#### exponential plot at the high overdensity end ####
# fig,ax = plt.subplots(1,1)

xlim0,xlim1 = -.8,1
ax2.set_xlim(0.,xlim1)
ax2.set_ylim(-1,-10)

ax2.set_xlabel('$\mathrm{log}(1+\delta)$')
ax2.set_ylabel('$\mathrm{log_{10}}(1 - \mathrm{CDF})$')

for i,temp in enumerate(mean_od):
    x = np.linspace(lo_lim[i],hi_lim[i])
    y = np.log10(1 - norm.cdf(x))
    ax2.fill_betweenx(y,xlim0,x,facecolor='C0',edgecolor="white",alpha=0.5)

ax2.scatter(print_df['log(1+delta)'],
           np.log10(1 - norm.cdf(print_df['log(1+delta)'])), s=15, color='black')

x = np.linspace(-.8,1)
ax2.plot(x,np.log10(1 - norm.cdf(x)), color='black', linestyle='dashed')

fig.savefig('CDF.png')

