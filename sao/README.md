## Simple Adaptive Optics (SAO)

SAO is a MATLAB code package to simulate simple AO stuffs.

- `sao_phasescreen.m` implements the phase screen method using subharmonic method [1].
- `sao_wfs_CodedWavefrontSensor.m` implements our coded wavefront sensor [2].
- `sao_wfs_CurvatureWavefrontSensor.m` implements the curvature wavefront sensor [3].
- `sao_wfs_shackhartmann.m` implements the Shack-Hartmann wavefront sensor [4].
- `imnoise_dB.m` is a function to add noise.

The functions are tests in `./tests`.

### Reference

[1] R. G. Lane, A. Glindemann, and J. C. Dainty. "Simulation of a Kolmogorov phase screen." *Waves in random media* 2.3 (1992): 209-224.

[2] C. Wang, D. Xiong, Q. Fu, and W. Heidrich. "Ultra-high resolution coded wavefront sensor." *Optics express* 25.12 (2017): 13736-13746.

[3] Francois Roddier. "Curvature sensing and compensation: a new concept in adaptive optics." *Applied Optics* 27.7 (1988): 1223-1225.

[4] B. C. Platt, and R. Shack. "History and principles of Shack-Hartmann wavefront sensing." *Journal of Refractive Surgery* 17.5 (2001): S573-S577.