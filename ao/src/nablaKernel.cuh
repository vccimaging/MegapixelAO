#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// texture references
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives with stencil in MATLAB form: [-1 1 0]
///
/// CUDA kernel, relies heavily on texture unit
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[out] Ix      x derivative
/// \param[out] Iy      y derivative
///////////////////////////////////////////////////////////////////////////////
__global__ void nablaKernel(const float *I, int width, int height, float *Ix, float *Iy)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * width;

    if (ix >= width || iy >= height) return;
    
    // replicate boundary condition
    if(ix == 0){
        Ix[pos] = 0.0f;
    } 
    else{
        Ix[pos] = I[pos] - I[(ix-1) + iy * width];
    }
    if (iy == 0){
        Iy[pos] = 0.0f;
    }
    else{
        Iy[pos] = I[pos] - I[ix     + (iy-1) * width];  
    }
}

///////////////////////////////////////////////////////////////////////////////
/// \brief compute image derivatives
///
/// \param[in]  I        input image
/// \param[in]  w        image width
/// \param[in]  h        image height
/// \param[out] nabla_x  x derivative
/// \param[out] nabla_y  y derivative
///////////////////////////////////////////////////////////////////////////////
static
void nabla(const float *I, int w, int h, float *nabla_x, float *nabla_y)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    nablaKernel<<<blocks, threads>>>(I, w, h, nabla_x, nabla_y);
}
