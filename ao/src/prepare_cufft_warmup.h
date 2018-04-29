#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <helper_cuda.h>

typedef cufftComplex complex;

#ifndef PREPARE_CUFFT_WARMUP_H
#define PREPARE_CUFFT_WARMUP_H

void cufft_warper(complex *h_in, int n, int m, cufftHandle plan, complex *h_out);

#endif
