# ic_selection
Select regions for zoom-ICs. Requires a density grid generated with [DensityGridder](https://github.com/christopherlovell/densitygridder)

## Files


- `convolve.py` loads the gridded density field in `grid_weights/`, and smoothes with a given top-hat convolution window.

- `fit_and_select` fits the overdensity distribution (and plots it), and provides methods for selecting based on overdensity and distance from the mean.

- `methods.py` contains methods for exporting files, as well as the `simulation` class containing information on the source simulation boxsize, the grid size, and the smoothing window. All changes to these parameters should be made here to ensure consistency across scripts.

## Instructions
Run the following scripts in the following order:
```
python convolve.py
python fit_and_select.py
```
