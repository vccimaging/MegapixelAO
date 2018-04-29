#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// texture references
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// \brief compute divergence with gradient stencil in MATLAB form: [-1 1 0]
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[out] div     divergence
///////////////////////////////////////////////////////////////////////////////
__global__ void nablaTKernel(const float *Ix, const float *Iy, 
                             int width, int height, float *div)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * width;

    float val1, val2;
    if (ix >= width || iy >= height) return;
    
    // replicate boundary condition
    if(ix == width-1){
        val1 = 0.0f;
    } 
    else{
        val1 = Ix[pos] - Ix[ix+1  + iy     * width];
    }
    if (iy == height-1){
        val2 = 0.0f;
    }
    else{
        val2 = Iy[pos] - Iy[ix    + (iy+1) * width];
    }
    div[pos] = val1 + val2;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute divergence
///
/// \param[in]  Ix       input grad_x
/// \param[in]  Iy       input grad_y
/// \param[in]  w        image width
/// \param[in]  h        image height
/// \param[out] div      divergence
///////////////////////////////////////////////////////////////////////////////
static
void nablaT(const float *Ix, const float *Iy, 
            int w, int h, float *div)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    nablaTKernel<<<blocks, threads>>>(Ix, Iy, w, h, div);
}
