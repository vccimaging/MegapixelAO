#include <opencv2/highgui.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>

// for thrust functions
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <cmath>

#include "gen_SLM_images.h"

template <typename T>
struct xfun
{
    __device__
        T operator()(const T& x) const { 
            return x;
        }
};

template <typename T>
struct absfun2
{
    __device__
        T operator()(const T& x) const { 
            return fabs(x)*fabs(x);
        }
};

static __device__ float mod(float x)
{
	return x - floor(x + 0.5f) + 0.5f;
}

static
__global__ void updateSLM(float *SLM_phase, float *phi, int width, int height,
						  float scale_factor, float AO_gain, float mean)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height) return;

    const int pos = ix + iy*width;

    // get scale, and transform G0 y_k to \alpha G0 y_k
    float temp = AO_gain * scale_factor * (phi[pos] - mean);
	
	// update SLM as u_{k+1} = u_k + \alpha G0 y_k
	// (note: u_k is always zero-mean)
	SLM_phase[pos] = SLM_phase[pos] + temp;
}


static
__global__ void SLM2pi(unsigned char *SLM_phase, float *in, int width, int height, float mean)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height) return;

    const int pos = ix + iy*width;

	float temp = in[pos] - mean;
	SLM_phase[pos] = (unsigned char) ( mod(temp) * 255.0f );
}


extern
void gen_SLM_img(unsigned char *SLM_phase, 
				 cv::cuda::GpuMat phi_gpumat, cv::cuda::GpuMat temp_gpumat, cv::cuda::GpuMat SLM_gpumat,  
				 int width, int height, int SLM_width, int SLM_height, float scale_factor, float AO_gain, int kernel_size, 
				 cv::Mat G0, float& RMS_SLM, int BOUND_COND)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(SLM_width, threads.x), iDivUp(SLM_height, threads.y));

	// thrust setup arguments
    thrust::device_ptr<float> d_ptr;
    xfun<float>         unary_op;
    absfun2<float>      unary_op2;
    thrust::plus<float> binary_op;

	float mean_phi;
	
	// get mean of G0 y_k
	d_ptr = thrust::device_pointer_cast(phi_gpumat.ptr<float>(0));
    mean_phi = thrust::transform_reduce(d_ptr, d_ptr + width*height, unary_op, 0.0f, binary_op); 
    mean_phi = mean_phi / (float)(width*height);

	// map to the SLM plane: get G0 y_k
	cv::cuda::warpPerspective(phi_gpumat, temp_gpumat, G0, temp_gpumat.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
//	cv::cuda::warpPerspective(phi_gpumat, temp_gpumat, G0, temp_gpumat.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT);

	// update SLM as u_{k+1} = u_k + \alpha G0 y_k
	// The rule: u_k is always of mean 127; \alpha G0 y_k should normalize to [0 255] with a mean 127
    updateSLM<<<blocks, threads>>>(SLM_gpumat.ptr<float>(0), temp_gpumat.ptr<float>(0), SLM_width, SLM_height, scale_factor, AO_gain, mean_phi);

	// calculate the mean of u_{k+1}
    d_ptr = thrust::device_pointer_cast(SLM_gpumat.ptr<float>(0));
    mean_phi = thrust::transform_reduce(d_ptr, d_ptr + SLM_width*SLM_height, unary_op, 0.0f, binary_op); 
    mean_phi = mean_phi / (float)(SLM_width*SLM_height);
	
	// set bilateral filter parameters
	float sigma_pixel = 50.0f;
	float sigma_space = 50.0f;
	
	// bilateral filtering on u_{k+1}; reduce noise and for better wrapping results
	switch (BOUND_COND){
		case 0:{
			cv::cuda::bilateralFilter(SLM_gpumat, SLM_gpumat, kernel_size, sigma_pixel, sigma_space, cv::BORDER_CONSTANT);
			break;
		}
		case 1:{
			cv::cuda::bilateralFilter(SLM_gpumat, SLM_gpumat, kernel_size, sigma_pixel, sigma_space, cv::BORDER_REPLICATE);
			break;
		}
	}
	
	// normalize it to zero-mean, and then transfer to the SLM (from [0 1] to [0 255])
    SLM2pi<<<blocks, threads>>>(SLM_phase, SLM_gpumat.ptr<float>(0), SLM_width, SLM_height, mean_phi);
    
    // calculate the SLM RMS
    d_ptr = thrust::device_pointer_cast(SLM_gpumat.ptr<float>(0));
    RMS_SLM = thrust::transform_reduce(d_ptr, d_ptr + SLM_width*SLM_height, unary_op2, 0.0f, binary_op);
    RMS_SLM = sqrtf( RMS_SLM / (float)(SLM_width*SLM_height) );
}


static
__global__ void test_SLM(float *SLM_phase, float *phi, int width, int height, float scale_factor, float AO_gain)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height) return;

    const int pos = ix + iy*width;

	SLM_phase[pos] += AO_gain * scale_factor * phi[pos];
}

extern
void test_acc_SLM(float *SLM_phase, float *phi, int width, int height, float scale_factor, float AO_gain)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

	// get the closed loop total phase
    test_SLM<<<blocks, threads>>>(SLM_phase, phi, width, height, scale_factor, AO_gain);
}
