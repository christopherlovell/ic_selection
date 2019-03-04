import numpy as np

class simulation():

    def __init__(self):
        self.h = 0.6777
        self.boxsize = 3200
        self.r = 10. / self.h
        self.grid = 1200
        self.conv = (self.boxsize/self.grid)

    def show(self):
        print('Hubble parameter:',self.h)
        print('Boxsize (cMpc):',self.boxsize)
        print('Grid dimensions:',self.grid,'^3')



def write_multidimensional_array(arr, fname):
    """
    see https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file

    load with `np.loadtxt(fname)`
    """

    # Write the array to disk
    with open(fname, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(arr.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in arr:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

    return True


def near_coords(pos, r, conv):
    
    """
    Gives the coordinates of the points near to the given point inside a a given radius
    coords: in terms of the simuation box coordinates
    pos_coords: array position
    """
    
    a, b, c = pos
    pos_coords = []
    sz = int(max(r)/conv)+1
    
    x = np.arange(a-sz, a+sz+1)
    y = np.arange(b-sz, b+sz+1)
    z = np.arange(c-sz, c+sz+1)
    
    x[x < 0] += grid
    y[y < 0] += grid
    z[z < 0] += grid
    
    x[x >= grid] -= grid
    y[y >= grid] -= grid
    z[z >= grid] -= grid
    
    for i in x: 
        for j in y:
            for k in z:
                pos_coords.append([i, j, k])
    
    pos_coords = np.array(pos_coords)
    
    coords = pos_coords.astype(float)*conv 
    
    return coords, pos_coords


