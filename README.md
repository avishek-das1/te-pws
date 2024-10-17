# Transfer Entropy- Path Weight Sampling
Curated code and data for reproducing the findings in _Exact computation of transfer entropy with Path Weight Sampling_, Avishek Das & Pieter Rein ten Wolde, [arxiv:2409.01650](https://arxiv.org/abs/2409.01650) (2024).

The curated codes are for computing transfer entropies with the TE-PWS algorithm in a 3D Langevin model (`TEPWS_OU_3D_transferentropies.py`) and a 3D Chemical Reaction Network (`TEPWS_jump_3D_transferentropies.py`). The code imports the libraries numpy, math, sys and numba, so they must first be installed in the python version of your system. The code then runs in parallel using all available cores.
For running the codes from the command line, call:
```
python TEPWS_OU_3D_transferentropies.py n
```
or
```
python TEPWS_jump_3D_transferentropies.py n
```
where n should be replaced with an integer to seed the random number generator.

The directory `figures/` has all data and scripts required to produce the three figures in the main text and the figure in the Supplemental Material. Each figure can be plotted by entering the corresponding directory, say `figures/fig1/`, and calling the corresponding python script from the command line, such as:
```
python plot_schematic.py
```
This will load the required data, create the figure and save it as a `.png` file, such as `fig_schematic.png`.
