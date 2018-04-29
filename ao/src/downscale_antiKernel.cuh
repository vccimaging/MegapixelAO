#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// texture references
///////////////////////////////////////////////////////////////////////////////

/// image to downscale
extern texture<float, 2, cudaReadModeElementType> texFine;

static
__global__ void anti_weight_x_Kernel(int width, int height, int stride, float *out)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height)
        return;

    float dx = 1.0f/(float)width;
    float dy = 1.0f/(float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

	x -= 0.25*dx;
	y -= 0.25*dy;
	
    dx /= 2.0f;
    dy /= 2.0f;
    
    // magic number, in MATLAB: conv2([0.125 0.375 0.375 0.125],[0.125 0.375 0.375 0.125]')
	out[ix + iy * stride] = 
	0.015625f*tex2D(texFine, x-dx, y-1*dy) + 0.046875f*tex2D(texFine, x, y-1*dy) + 0.046875f*tex2D(texFine, x+dx, y-1*dy) + 0.015625f*tex2D(texFine, x+2*dx, y-1*dy) +
    0.046875f*tex2D(texFine, x-dx, y     ) + 0.140625f*tex2D(texFine, x, y     ) + 0.140625f*tex2D(texFine, x+dx, y     ) + 0.046875f*tex2D(texFine, x+2*dx, y     ) +
    0.046875f*tex2D(texFine, x-dx, y+1*dy) + 0.140625f*tex2D(texFine, x, y+1*dy) + 0.140625f*tex2D(texFine, x+dx, y+1*dy) + 0.046875f*tex2D(texFine, x+2*dx, y+1*dy) +
    0.015625f*tex2D(texFine, x-dx, y+2*dy) + 0.046875f*tex2D(texFine, x, y+2*dy) + 0.046875f*tex2D(texFine, x+dx, y+2*dy) + 0.015625f*tex2D(texFine, x+2*dx, y+2*dy);
}


///////////////////////////////////////////////////////////////////////////////
/// \brief Downscale kernel
///
/// \param[in]  src          source image
/// \param[in]  width        width
/// \param[in]  height       height
/// \param[in]  stride       stride
/// \param[in]  newWidth     new width (after downscaling)
/// \param[in]  newHeight    new height (after downscaling)
/// \param[in]  newStride    new stride (after downscaling)
/// \param[out] out          output downscaled image
///////////////////////////////////////////////////////////////////////////////
extern
void Downscale_Anti(const float *src, int width, int height, int stride,
               int newWidth, int newHeight, int newStride, float *out)
{
    dim3 threads(32, 8);
    dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

    // mirror if a coordinate value is out-of-range
    texFine.addressMode[0] = cudaAddressModeMirror;
    texFine.addressMode[1] = cudaAddressModeMirror;
    texFine.filterMode = cudaFilterModeLinear;
    texFine.normalized = true;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    checkCudaErrors(cudaBindTexture2D(0, texFine, src, width, height, stride * sizeof(float)));

    anti_weight_x_Kernel<<<blocks, threads>>>(newWidth, newHeight, newStride, out);
}

