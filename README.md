# Transfer Entropy- Path Weight Sampling
Curated code for Transfer Entropy- Path Weight Sampling for computing transfer entropies in Langevin models. As described in 'Computing exact transfer entropy with Path Weight Sampling', Avishek Das & Pieter Rein ten Wolde, AMOLF, Amsterdam (2024).

Requires importing numpy, math, sys and numba.
Automatically runs in parallel and uses all the cores in your system.
For running call:
python TEPWS_OU_3D_transferentropies.py n
where n should be an integer to seed the random number generator.
