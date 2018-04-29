#include "common.h"

__global__ void fista_iter(float *p_x, int width, int height, float *b, float theta)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * width;

    if (ix >= width || iy >= height) return;
    
    p_x[pos] = p_x[pos] - b[pos] / theta;	
}

__global__ void ScaleImageKernel(float *img, float *IMG, int width, int height, 
								 float vlow, float vhigh, float ilow, float ihigh)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * width;

    if (ix >= width || iy >= height) return;
    
	IMG[pos] = (img[pos] - ilow) / (ihigh - ilow) * (vhigh - vlow) + vlow;
}


__global__ void prox_LinfKernel(float *p, float *temp, int width, int height, 
							    float theta, float tau)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * width;

    if (ix >= width || iy >= height) return;
    
	p[pos] = min(max(p[pos] - tau*temp[pos], -1.0f), 1.0f);
}


__global__ void primalKernel(float *im, float *IM, int width, int height, 
							 float theta, float alp, float *temp)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * width;

    if (ix >= width || iy >= height) return;
    
    // IM = IM - nablaT(p) * theta;
    // im = im - alp*IM
	im[pos] = im[pos] - alp*(IM[pos] - temp[pos]*theta);
}

