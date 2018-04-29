#include <cufft.h>

#include "common.h"

typedef float real;
typedef cufftComplex complex;

#define PI 3.1415926535897932384626433832795028841971693993751


__global__
void computeDCTweightsKernel(int N_height, complex *out)
{
    const int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos >= N_height) return;

    // we use type double to enhance accuracy
    out[pos].x = (float) (sqrt((double)(2*N_height)) * cos(pos*PI/(2*N_height)));
    out[pos].y = (float) (sqrt((double)(2*N_height)) * sin(pos*PI/(2*N_height)));
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute DCT weights
/// \param[in]  N_width     unknown width
/// \param[in]  N_height    unknown height
/// \param[out] ww_1        result (dimension along width)
/// \param[out] ww_2        ...... (dimension along height)
///////////////////////////////////////////////////////////////////////////////
static
void computeDCTweights(int N_width, int N_height, complex *ww_1, complex *ww_2)
{
    dim3 threads(256);
    dim3 blocks1(iDivUp(N_height, threads.x));
    dim3 blocks2(iDivUp(N_width,  threads.x));
    
    computeDCTweightsKernel<<<blocks1, threads>>>(N_height, ww_1);
    computeDCTweightsKernel<<<blocks2, threads>>>(N_width,  ww_2);
}
