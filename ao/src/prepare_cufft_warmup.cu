#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <helper_cuda.h>
#include <assert.h>

typedef cufftComplex complex;

void cufft_warper(complex *h_in, int n, int m, cufftHandle plan, complex *h_out)
{
    const int data_size = n*m*sizeof(complex);

    // device memory allocation
    complex *d_temp;
    checkCudaErrors(cudaMalloc(&d_temp,  data_size));

    // transfer data from host to device
    checkCudaErrors(cudaMemcpy(d_temp, h_in, data_size, cudaMemcpyHostToDevice));

	// Compute the FFT
	cufftExecC2C(plan, d_temp, d_temp, CUFFT_FORWARD);

    // transfer result from device to host
    checkCudaErrors(cudaMemcpy(h_out, d_temp, data_size, cudaMemcpyDeviceToHost));

    // cleanup
    checkCudaErrors(cudaFree(d_temp));
}