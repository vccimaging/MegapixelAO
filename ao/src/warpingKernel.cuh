#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// texture references
///////////////////////////////////////////////////////////////////////////////

/// image to warp
texture<float, 2, cudaReadModeElementType> texToWarp;

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
    //    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__
float w1(float a)
{
    //    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
    //    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
    return -1.0f + w1(a) / (w0(a) + w1(a));
}

__device__ float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a));
}

// filter 4 values using cubic splines
__device__
float cubicFilter(float x, float c0, float c1, float c2, float c3)
{
    float r = c0 * w0(x) + c1 * w1(x) + c2 * w2(x) + c3 * w3(x);
    return r;
}


__device__
float tex2DBicubic(const texture<float, 2, cudaReadModeElementType> texref, float x, float y)
{
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

	// shift half pixel for the texture memory coordinate
	px += 0.5f;
	py += 0.5f;

    // fast but imprecise bicubic lookup using 4 texture lookups
    // note: we could store these functions in a lookup table texture, but maths is cheap
//    float g0x = g0(fx);
//    float g1x = g1(fx);
//    float h0x = h0(fx);
//    float h1x = h1(fx);
//    float h0y = h0(fy);
//    float h1y = h1(fy);

//    return g0(fy) * (g0x * tex2D(texref, px + h0x, py + h0y)  +
//                     g1x * tex2D(texref, px + h1x, py + h0y)) +
//           g1(fy) * (g0x * tex2D(texref, px + h0x, py + h1y)  +
//                     g1x * tex2D(texref, px + h1x, py + h1y));

    // slow but precise bicubic lookup using 16 texture lookups
    return cubicFilter(fy,
           cubicFilter(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2, py-1)),
           cubicFilter(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
           cubicFilter(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
           cubicFilter(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2)));
}


///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with a given displacement field, CUDA kernel.
/// \param[in]  width   image width
/// \param[in]  height  image height
/// \param[in]  u       horizontal displacement
/// \param[in]  v       vertical displacement
/// \param[out] out     result
///////////////////////////////////////////////////////////////////////////////
__global__ void WarpingKernel(int N_width, int N_height,
                              int M_width, int M_height, int M_stride,
                              const float *u, const float *v, float *out)
{
    const int L_width  = (N_width  - M_width) /2;
    const int L_height = (N_height - M_height)/2;
    
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos_M = ix + iy * M_stride;
    const int pos_N = (ix+L_width) + (iy+L_height) * N_width;
    
    if (ix >= M_width || iy >= M_height) return;

    // warp on reference image I0:
    float x = (float)ix - u[pos_N];
    float y = (float)iy - v[pos_N];

    out[pos_M] = tex2DBicubic(texToWarp, x, y);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief warp image with provided vector field, CUDA kernel wrapper.
///
/// For each output pixel there is a vector which tells which pixel
/// from a source image should be mapped to this particular output
/// pixel.
/// It is assumed that images and the vector field have the same stride and
/// resolution.
/// \param[in]  src source image
/// \param[in]  w   width
/// \param[in]  h   height
/// \param[in]  u   horizontal displacement
/// \param[in]  v   vertical displacement
/// \param[out] out warped image
///////////////////////////////////////////////////////////////////////////////
static
void WarpImage(float *src, int N_w, int N_h, int M_w, int M_h, int M_s,
               float *u, float *v, float *out)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(M_w, threads.x), iDivUp(M_h, threads.y));

    // zero if a coordinate value is out-of-range 
    // (it should be aligned with boundary condition in cubic spline coefficient computations)
    texToWarp.addressMode[0] = cudaAddressModeBorder;
    texToWarp.addressMode[1] = cudaAddressModeBorder;
    texToWarp.filterMode = cudaFilterModeLinear;
    texToWarp.normalized = false;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaBindTexture2D(0, texToWarp, src, M_w, M_h, M_s * sizeof(float));

    WarpingKernel<<<blocks, threads>>>(N_w, N_h, M_w, M_h, M_s, u, v, out);
}
