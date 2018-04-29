#include "common.h"

#include <helper_functions.h>

// include kernels
#include "computemat_x_hatKernel.cuh"
#include "computeDCTweightsKernel.cuh"

// for thrust functions
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <cmath>

///////////////////////////////////////////////////////////////////////////////
/// texture references
///////////////////////////////////////////////////////////////////////////////

/// image to downscale
texture<float, 2, cudaReadModeElementType> texFine;

static 
__global__
void transposeKernel(float *in, int width, int height, int stride1, int stride2, float *out)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= width || iy >= height) return;
    
    out[iy + ix * stride2] = in[ix + iy * stride1];
}


static
__device__ void tridisolve(float *x, float *b, float *mu, int m)
{
    // forward
    for (int i = 0; i < m-1; i++)
        x[i+1] -= mu[i] * x[i];
    
    // backward
    x[m-1] /= b[m-1];
    for (int i = m-2; i >= 0; i--)
        x[i] = (x[i] - x[i+1]) / b[i];
        
    // maginify by 6
    for (int i = 0; i < m; i++)
        x[i] = 6.0f * x[i];
}


static 
__global__ 
void tridisolve_parallel(float *img, float *b_h, float *mu_h, 
                           int width, int height, int stride2)
{
    const int iy = threadIdx.x + blockIdx.x * blockDim.x;

    if (iy >= width) return;
    
    tridisolve(img + iy * stride2, b_h, mu_h, height);
}


void cbanal(float *in, float *out, int width, int height, int stride1, int stride2)
{
    float *b  = new float [height];
    float *mu = new float [height-1];
    
    // set b
    for (int j = 0; j < height; j++)
        b[j] = 4.0f;

    // pre-computations
    for (int j = 0; j < height-1; j++)
    {
        mu[j] = 1.0f / b[j];
        b[j+1] -= mu[j];
    }
    
    float *d_b, *d_mu;
    checkCudaErrors(cudaMalloc(&d_b,  height*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mu, (height-1)*sizeof(float)));
    
    checkCudaErrors(cudaMemcpy(d_b, b, height*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mu, mu, (height-1)*sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 threads_1D(256);
    dim3 blocks_1D(iDivUp(width, threads_1D.x));
    
    dim3 threads_2D(32, 6);
    dim3 blocks_2D(iDivUp(width, threads_2D.x), iDivUp(height, threads_2D.y));
    
    transposeKernel<<<blocks_2D, threads_2D>>>(in, width, height, stride1, stride2, out);
    tridisolve_parallel<<<blocks_1D, threads_1D>>>(out, d_b, d_mu, width, height, stride2);

    // cleanup
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_mu));
    delete[] b;
    delete[] mu;
}



void cbanal2D(float *img, int width, int height, int stride1, int stride2)
{
    float *temp;    
    checkCudaErrors(cudaMalloc(&temp, height*width*sizeof(float)));
    
    // compute the cubic B-spline coefficients
    cbanal(img, temp, width, height, stride1, stride2);
    cbanal(temp, img, height, width, stride2, stride1);

    // cleanup    
    checkCudaErrors(cudaFree(temp));
}



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




///////////////////////////////////////////////////////////////////////////////
/// \brief Prepare pre-computation constants
///
/// handles memory allocations, control flow
/// \param[in]  N_W          unknown phase width
/// \param[in]  N_H          unknown phase height
/// \param[in]  alpha        degree of displacement field smoothness
/// \param[in]  mu           proximal parameter
/// \param[out] mat_x_hat    pre-computed mat_x_hat
/// \param[out] ww_1         ww_1 coefficient
/// \param[out] ww_2         ww_2 coefficient
///////////////////////////////////////////////////////////////////////////////
void prepare_precomputations(int N_W, int N_H, int currentLevel,
                             int *pW_N, int *pH_N, int *pS_N,
                             int *pW_M, int *pH_M, int *pS_M,
                             int *pW_L, int *pH_L,
                             float **pI0, float **pI1, float **d_I0_coeff,
                             float alpha, float *mu, const float **mat_x_hat,
                             const complex **ww_1, const complex **ww_2)
{
    printf("Pre-compute the variables on GPU...\n");

    if (currentLevel != 0){ // prepare pyramid
        for (; currentLevel > 0; currentLevel--)
        {
            int nw = pW_M[currentLevel] / 2;
            int nh = pH_M[currentLevel] / 2;
            int ns = iAlignUp(nw);
        
            pW_N[currentLevel - 1] = nw + 2*pW_L[currentLevel-1];
            pH_N[currentLevel - 1] = nh + 2*pH_L[currentLevel-1];
            pS_N[currentLevel - 1] = iAlignUp(pW_N[currentLevel - 1]);

            // pre-calculate mat_x_hat and store it
            checkCudaErrors(cudaMalloc(mat_x_hat + currentLevel, 
                                       pW_N[currentLevel] * pH_N[currentLevel] * sizeof(float)));
            computemat_x_hat(mu[currentLevel], alpha, pW_N[currentLevel], pH_N[currentLevel], 
                             (float *)mat_x_hat[currentLevel]);
        
            // pre-compute DCT weights and store them in device
            checkCudaErrors(cudaMalloc(ww_1 + currentLevel, pH_N[currentLevel]*sizeof(complex)));
            checkCudaErrors(cudaMalloc(ww_2 + currentLevel, pW_N[currentLevel]*sizeof(complex)));
            computeDCTweights(pW_N[currentLevel], pH_N[currentLevel], 
                              (complex *)ww_1[currentLevel], (complex *)ww_2[currentLevel]);

            checkCudaErrors(cudaMalloc(pI0 + currentLevel - 1, ns * nh * sizeof(float)));
            checkCudaErrors(cudaMalloc(pI1 + currentLevel - 1, ns * nh * sizeof(float)));
            checkCudaErrors(cudaMalloc(d_I0_coeff + currentLevel - 1, ns * nh * sizeof(float)));

			// downscale
            Downscale_Anti(pI0[currentLevel], pW_M[currentLevel], pH_M[currentLevel], pS_M[currentLevel], 
                        nw, nh, ns, (float *)pI0[currentLevel - 1]);
            Downscale_Anti(d_I0_coeff[currentLevel], pW_M[currentLevel], pH_M[currentLevel], pS_M[currentLevel], 
                        nw, nh, ns, (float *)d_I0_coeff[currentLevel - 1]);

            // pre-compute cubic coefficients and store them in device
            cbanal2D(d_I0_coeff[currentLevel], pW_M[currentLevel], pH_M[currentLevel], 
                                               pS_M[currentLevel], pH_M[currentLevel]);
    
            pW_M[currentLevel - 1] = nw;
            pH_M[currentLevel - 1] = nh;
            pS_M[currentLevel - 1] = ns;
        
            printf("pW_M[%d] = %d, pH_M[%d] = %d, pS_M[%d] = %d\n", 
                    currentLevel, pW_M[currentLevel], 
                    currentLevel, pH_M[currentLevel], 
                    currentLevel, pS_M[currentLevel]);
            printf("pW_N[%d] = %d, pH_N[%d] = %d, pS_N[%d] = %d\n", 
                    currentLevel, pW_N[currentLevel], 
                    currentLevel, pH_N[currentLevel], 
                    currentLevel, pS_N[currentLevel]);
            printf("pW_L[%d] = %d, pH_L[%d] = %d\n", 
                    currentLevel, pW_L[currentLevel], 
                    currentLevel, pH_L[currentLevel]);
        }
    }
    
    // pre-calculate mat_x_hat and store it
    checkCudaErrors(cudaMalloc(mat_x_hat + currentLevel, 
                               pW_N[currentLevel] * pH_N[currentLevel] * sizeof(float)));
    computemat_x_hat(mu[currentLevel], alpha, pW_N[currentLevel], pH_N[currentLevel], (float *)mat_x_hat[currentLevel]);
    
    // pre-compute DCT weights and store them in device
    checkCudaErrors(cudaMalloc(ww_1 + currentLevel, pH_N[currentLevel]*sizeof(complex)));
    checkCudaErrors(cudaMalloc(ww_2 + currentLevel, pW_N[currentLevel]*sizeof(complex)));
    computeDCTweights(pW_N[currentLevel], pH_N[currentLevel], 
                      (complex *)ww_1[currentLevel], (complex *)ww_2[currentLevel]);
    
    // pre-compute cubic coefficients and store them in device
    cbanal2D(d_I0_coeff[currentLevel], pW_M[currentLevel], pH_M[currentLevel], 
                                       pS_M[currentLevel], pH_M[currentLevel]);
            
    printf("pW_M[%d] = %d, pH_M[%d] = %d, pS_M[%d] = %d\n", 
            currentLevel, pW_M[currentLevel], 
            currentLevel, pH_M[currentLevel], 
            currentLevel, pS_M[currentLevel]);
    printf("pW_N[%d] = %d, pH_N[%d] = %d, pS_N[%d] = %d\n", 
            currentLevel, pW_N[currentLevel], 
            currentLevel, pH_N[currentLevel], 
            currentLevel, pS_N[currentLevel]);
    printf("pW_L[%d] = %d, pH_L[%d] = %d\n", 
            currentLevel, pW_L[currentLevel], 
            currentLevel, pH_L[currentLevel]);
}


