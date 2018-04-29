#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// texture references
///////////////////////////////////////////////////////////////////////////////

/// scalar field to upscale
texture<float, 2, cudaReadModeElementType> texCoarse;

static
__global__ void UpscaleKernel(int width, int height, float scale, float *out)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height) return;

    float x = ((float)ix + 0.5f) / (float)width;
    float y = ((float)iy + 0.5f) / (float)height;

    // exploit hardware interpolation
    // and scale interpolated vector to match next pyramid level resolution
    out[ix + iy * width] = tex2D(texCoarse, x, y) * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief upscale one component of a displacement field, kernel wrapper
/// \param[in]  src         field component to upscale
/// \param[in]  width       field current width
/// \param[in]  height      field current height
/// \param[in]  newWidth    field new width
/// \param[in]  newHeight   field new height
/// \param[in]  scale       value scale factor (multiplier)
/// \param[out] out         upscaled field component
///////////////////////////////////////////////////////////////////////////////
static
void Upscale(const float *src, int width, int height,
             int newWidth, int newHeight, float scale, float *out)
{
    dim3 threads(32, 8);
    dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

    // mirror if a coordinate value is out-of-range
    texCoarse.addressMode[0] = cudaAddressModeMirror;
    texCoarse.addressMode[1] = cudaAddressModeMirror;
    texCoarse.filterMode = cudaFilterModeLinear;
    texCoarse.normalized = true;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaBindTexture2D(0, texCoarse, src, width, height, width * sizeof(float));

    UpscaleKernel<<<blocks, threads>>>(newWidth, newHeight, scale, out);
}
