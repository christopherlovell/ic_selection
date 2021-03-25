# ic_selection
Select regions for zoom-ICs. Requires a density grid generated with [DensityGridder](https://github.com/christopherlovell/densitygridder)

## Files

- `sim_test.py` -> an example simulation file. Contains a class detailing the simulation. Copy and edit this for each simulation.

- `convolve.py` loads the specified gridded density field, and smoothes with a top-hat convolution window (specified in the simulation file). Called as follows:

`python convolve.py sim_test grid_weights/example_grid_file.dat output/example_output_file.dat`

- `fit.py` fits the overdensity distribution (and plots it). Must specify simulation file as argument.

- `select_regions.py` provides methods for selecting based on overdensity and distance from the mean. It first selects the 15 highest overdensities, so that the subsequent ones don’t overlap with it. Accepts the required sigma of the region as an argument (after the specified simulation file) and displays a subset of the selected coordinates.

- `methods.py` contains methods for exporting files, as well as the `simulation` class containing information on the source simulation boxsize, the grid size, and the smoothing window. All changes to these parameters should be made here to ensure consistency across scripts.

- `weights_grid.py` use this for generating the weights of the resim regions 

## Instructions
Run the following scripts in the following order:
```
python convolve.py
python fit.py
python select_regions.py
```
