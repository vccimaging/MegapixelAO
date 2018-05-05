## Megapixel Adaptive Optics
This is the open source repository for our paper to appear in SIGGRAPH 2018:

[**Megapixel Adaptive Optics: Towards Correcting Large-scale Distortions in Computational Cameras**](http://vccimaging.org/Publications/Wang2018AdaptiveOptics/)

[Congli Wang](https://congliwang.github.io), [Qiang Fu](http://vccimaging.org/People/fuq/), [Xiong Dun](http://vccimaging.org/People/dunx/), and [Wolfgang Heidrich](http://vccimaging.org/People/heidriw/)

King Abdullah University of Science and Technology (KAUST)

### Overview

This repository contains:

- An improved version of the wavefront solver described in [Ultra-high resolution coded wavefront sensor](https://www.osapublishing.org/abstract.cfm?uri=oe-25-12-13736) in MATLAB and CUDA:
  - Linearized version, solver is either ADMM or conjugate gradient;
  - Pyramid version.
- Adaptive optics (AO) control code in C++ and CUDA:

  - The AO control logic described in the paper;
  - SLM instant control code via CUDA-OpenGL interop.
- Wave optics simulation code in MATLAB:

  - Simulation based on Rayleigh-Sommerfeld diffraction formula;
  - Simulation and solver for Shack-Hartmann sensor, curvature sensor (TIE based), and our coded wavefront sensor.
- 3D printed model in Solidworks.

### Prerequisite

#### Hardware

See the Supplementary Material for a full hardware list. For the AO code to successfully run, it requires your computer to be connected with:

- PointGrey USB cameras. We used [GS3-U3-15S5M-C](https://www.ptgrey.com/grasshopper3-14-mp-mono-usb3-vision-sony-icx825-2).
- Spatial light modulators (SLM). We used [Holoeye PLUTO](https://holoeye.com/spatial-light-modulators/slm-pluto-phase-only/).

Also your computer should feature with NVIDIA based graphic cards for CUDA to run.

#### Software

Our code has been tested on Ubuntu 16.04 and Windows 10. In theory it should also work on Mac OS X.

For MATLAB code, just plug & play. 

Please refer to `./ao/README.md` for a full installation guide on library dependency and how to compile the C++ and CUDA code.

### Citation

```
@article{wang2018megapixel,
  title = {Megapixel adaptive optics: towards correcting large-scale distortions in computational cameras},
  author = {Wang, Congli and Fu, Qiang and Dun, Xiong and Heidrich, Wolfgang},
  journal = {ACM Transactions on Graphics (Proc. SIGGRAPH)},
  volume = {37},
  number = {4},
  pages = {115},
  year = {2018},
  publisher={ACM}
}
```

```
@article{wang2017ultra,
  title={Ultra-high resolution coded wavefront sensor},
  author={Wang, Congli and Dun, Xiong and Fu, Qiang and Heidrich, Wolfgang},
  journal={Optics express},
  volume={25},
  number={12},
  pages={13736--13746},
  year={2017},
  publisher={Optical Society of America}
}
```

### Contact

We welcome any comments and questions. Please either open up an issue, or send email to congli.wang@kaust.edu.sa.

