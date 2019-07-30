# ic_selection
Select regions for zoom-ICs. Requires a density grid generated with [DensityGridder](https://github.com/christopherlovell/densitygridder)

## Files


- `convolve.py` loads the gridded density field in `grid_weights/`, and smoothes with a given top-hat convolution window.

- `fit.py` fits the overdensity distribution (and plots it)

- `select_regions.py` provides methods for selecting based on overdensity and distance from the mean. Accepts the required sigma of the region as an argument and displays a subset of the selected coordinates.

- `methods.py` contains methods for exporting files, as well as the `simulation` class containing information on the source simulation boxsize, the grid size, and the smoothing window. All changes to these parameters should be made here to ensure consistency across scripts.

## Instructions
Run the following scripts in the following order:
```
python convolve.py
python fit.py
python select_regions.py
```
