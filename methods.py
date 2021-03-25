import numpy as np

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


def grid_coordinates(coods, sim):
    """
    Find overdensity / sigma for given coordinates

    args:
        coods (array)
        sim (simulation object)
    """
    return (coods / sim.conv).astype(int)

