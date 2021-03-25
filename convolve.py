import os, sys
import importlib
import numpy as np
from scipy import ndimage

from methods import write_multidimensional_array

simname = sys.argv[1]
fname = sys.argv[2]
outname = sys.argv[3]

sim = importlib.import_module(simname)
sim = sim.simulation()
sim.show()

#dat = np.fromfile('./grid_weights/out_1200.bin',sep='\n')
dat = np.fromfile(fname,sep='\n')
dat = np.reshape(dat, (sim.grid,sim.grid,sim.grid), 'F')


def spherical_kernel(r,boxl,gridsize):
    """
    Defines a spherical kernel, radius `r`
    """
    dl = boxl / gridsize  
    sz = r/dl      # round up instead of down
    length = 2*round(sz)         

    # kernel defined from a given coordinate
    # so to get a symmetric kernel need to have odd length sides
    ker = np.zeros((length+1,length+1,length+1))
    
    x, y, z = (length)/2., (length)/2., (length)/2.
    
    sz_squared = sz**2  
    sel = np.arange(length+1)
    for i in sel:
        for j in sel:
            for k in sel:
                if (i-x)**2 + (j-y)**2 + (k-z)**2 <= sz_squared:
                    ker[i,j,k] = 1.
                
    return ker



kernel = spherical_kernel(sim.r,sim.boxsize,sim.grid)

## print kernel shape
for i in np.arange(kernel.shape[0]): print(kernel[6,i,:])

conv_grid = ndimage.convolve(dat, kernel, mode='wrap')

mean_density = np.mean(conv_grid)
delta = conv_grid/mean_density # delta + 1

#write_multidimensional_array(delta, 'conv_out.txt')
# np.savez_compressed('conv_output', delta = delta)
np.savez_compressed(outname, delta = delta)


