
import numpy as np
from scipy import ndimage

from methods import write_multidimensional_array,sim

sim = simulation()
sim.show()
snap = '002'

dat = np.fromfile('grid_weights/weights_3200_snap%s_res%s.dat'%(snap,sim.gridsize),sep='\n')
dat = np.reshape(dat, (sim.gridsize,sim.gridsize,sim.gridsize), 'F')


# @jit(float64(float64[:]))
def spherical_kernel(r,boxl,gridsize):
    """
    Defines a spherical kernel, radius `r`
    """
    # sz = int(r/conv)    # TODO: what is `conv`? why no argument?
    dl = boxl / gridsize  # define `dl` instead of `conv`, added `boxl` and `gridsize`args
    sz = r/dl      # round up instead of down
    # length = 2*sz+4     # TODO: why +4?
    length = 2*round(sz)         # remove +4

    # kernel defined from a given coordinate
    # so to get a symmetric kernel need to have odd length sides
    ker = np.zeros((length+1,length+1,length+1))
    
    x, y, z = (length)/2., (length)/2., (length)/2.
    
    sz_squared = sz**2  # do this before the loop
    sel = np.arange(length+1)
    for i in sel:
        for j in sel:
            for k in sel:
                # if (i-x)**2+(j-y)**2+(k-z)**2<=(r/conv)**2:
                if (i-x)**2 + (j-y)**2 + (k-z)**2 <= sz_squared:
                    ker[i,j,k] = 1.
                
    return ker



kernel = spherical_kernel(sim.r,sim.boxsize,sim.gridsize)

## print kernel shape
for i in np.arange(kernel.shape[0]): print(kernel[6,i,:])

conv_grid = ndimage.convolve(dat, kernel, mode='wrap')

mean_density = np.mean(conv_grid)
delta = conv_grid/mean_density # delta + 1

write_multidimensional_array(delta, 'conv_out.txt')

