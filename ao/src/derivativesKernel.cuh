#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// texture references
///////////////////////////////////////////////////////////////////////////////

/// source image
texture<float, 2, cudaReadModeElementType> texSource;
/// tracked image
texture<float, 2, cudaReadModeElementType> texTarget;

__global__ void ComputeDerivativesKernel(int width, int height, float mu,
                                         float *w11, float *w12_or_w22, float *w13, float *w21, float *w23)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * width;

    if (ix >= width || iy >= height) return;

    float dx = 1.0f / (float)width;
    float dy = 1.0f / (float)height;

    float x = ((float)ix + 0.5f) * dx;
    float y = ((float)iy + 0.5f) * dy;

    float gx, gy, gt;
    float t0, t1;
    // x derivative
    t0  = tex2D(texSource, x - 2.0f * dx, y);
    t0 -= tex2D(texSource, x - 1.0f * dx, y) * 8.0f;
    t0 += tex2D(texSource, x + 1.0f * dx, y) * 8.0f;
    t0 -= tex2D(texSource, x + 2.0f * dx, y);
    t0 /= 12.0f;

    t1  = tex2D(texTarget, x - 2.0f * dx, y);
    t1 -= tex2D(texTarget, x - 1.0f * dx, y) * 8.0f;
    t1 += tex2D(texTarget, x + 1.0f * dx, y) * 8.0f;
    t1 -= tex2D(texTarget, x + 2.0f * dx, y);
    t1 /= 12.0f;

    gx = (t0 + t1) * 0.5f;
    
    // t derivative
    gt = tex2D(texTarget, x, y) - tex2D(texSource, x, y);
    
    // y derivative
    t0  = tex2D(texSource, x, y - 2.0f * dy);
    t0 -= tex2D(texSource, x, y - 1.0f * dy) * 8.0f;
    t0 += tex2D(texSource, x, y + 1.0f * dy) * 8.0f;
    t0 -= tex2D(texSource, x, y + 2.0f * dy);
    t0 /= 12.0f;

    t1  = tex2D(texTarget, x, y - 2.0f * dy);
    t1 -= tex2D(texTarget, x, y - 1.0f * dy) * 8.0f;
    t1 += tex2D(texTarget, x, y + 1.0f * dy) * 8.0f;
    t1 -= tex2D(texTarget, x, y + 2.0f * dy);
    t1 /= 12.0f;

    gy = (t0 + t1) * 0.5f;
    
    float gxx = gx*gx;
    float gxy = gx*gy;
    float gyy = gy*gy;
    float denom = gxx + gyy + mu/2;
    
           w11[pos] = (mu/2 + gyy) / denom;
    w12_or_w22[pos] = - gxy / denom;
           w13[pos] = - gx*gt / denom;
           w21[pos] = (mu/2 + gxx) / denom;
           w23[pos] = - gy*gt / denom;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief Compute image derivatives (gx, gy, gt in the paper)
///
/// \param[in]  I0  source image
/// \param[in]  I1  tracked image
/// \param[in]  w   image width
/// \param[in]  h   image height
/// \param[in]  s   stride
/// \param[in]  mu  proximal parameter
/// \param[med] Ix  x derivative
/// \param[med] Iy  y derivative
/// \param[med] Iz  temporal derivative
/// \param[out] w11         pre-cached constants
/// \param[out] w12_or_w22  .
/// \param[out] w13         .
/// \param[out] w21         .
/// \param[out] w23         .
///////////////////////////////////////////////////////////////////////////////
static
void ComputeDerivatives(const float *I0, const float *I1,
                        int w, int h, int s, float mu,
                        float *w11, float *w12_or_w22, float *w13, float *w21, float *w23)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    // replicate if a coordinate value is out-of-range
    texSource.addressMode[0] = cudaAddressModeClamp;
    texSource.addressMode[1] = cudaAddressModeClamp;
    texSource.filterMode = cudaFilterModeLinear;
    texSource.normalized = true;

    texTarget.addressMode[0] = cudaAddressModeClamp;
    texTarget.addressMode[1] = cudaAddressModeClamp;
    texTarget.filterMode = cudaFilterModeLinear;
    texTarget.normalized = true;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaBindTexture2D(0, texSource, I0, w, h, s * sizeof(float));
    cudaBindTexture2D(0, texTarget, I1, w, h, s * sizeof(float));

    ComputeDerivativesKernel<<<blocks, threads>>>(w, h, mu, w11, w12_or_w22, w13, w21, w23);
}
