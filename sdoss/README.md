## SDOSS: Simple Diffraction Optics Simulation System  (SDOSS)

SDOSS is a MATLAB code package to implement the Rayleigh-Sommerfeld diffraction formula. SDOSS implements two types of method:

- Direct integration [1].
- Angular spectrum [2, 3].

In our paper, we used the angular spectrum method to produce most of the simulations.

Demos are:

- `demo_sdoss.m` demonstrates the basic usage of the package.
- `demo_theory_test.m` reproduces Figure 4 in paper [1].

### Reference

[1] Fabin Shen and Anbo Wang. "Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula." Applied optics 45.6 (2006): 1102-1110.

[2] Kyoji Matsushima and Tomoyoshi Shimobaba. "Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields." Optics express 17.22 (2009): 19662-19673.

[3] Joseph Goodman. "Introduction to Fourier optics." (2008).