#include "common.h"

#define PI 3.1415926535897932384626433832795028841971693993751

__global__
void mat_x_hatKernel(float mu, float alpha, int N_width, int N_height, 
                     float *mat_x_hat)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= N_width || iy >= N_height) return;

    const int pos = ix + iy * N_width;

    if (ix == 0 && iy == 0)
    {
        mat_x_hat[pos] = 1.0f;
    }
    else
    {
        // we use double precision here for better accuracy
        mat_x_hat[pos] = (float) ( -(mu+2*alpha) * ( 2*cos(PI*(double)ix/(double)N_width)  +
                                                     2*cos(PI*(double)iy/(double)N_height) - 4.0));
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute mat_x_hat
/// \param[in]  mu          proximal parameter
/// \param[in]  alpha       regularization parameter
/// \param[in]  M_width     unknown width
/// \param[in]  M_height    unknown height
/// \param[out] mat_x_hat   result
///////////////////////////////////////////////////////////////////////////////
static
void computemat_x_hat(float mu, float alpha, int N_width, int N_height, 
                      float *mat_x_hat)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(N_width, threads.x), iDivUp(N_height, threads.y));

    mat_x_hatKernel<<<blocks, threads>>>(mu, alpha, N_width, N_height, mat_x_hat);
}
