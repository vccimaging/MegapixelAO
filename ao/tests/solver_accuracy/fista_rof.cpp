#if defined(LINUX32) || defined(LINUX64)
#define LINUX
#endif

// Please keep this include order to make sure a successful compilation!
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <cuda_profiler_api.h> // for profiling
#include <helper_cuda.h> 

// OpenCV
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

// std
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#include <cufft.h>
typedef cufftComplex complex;

// our library
#include "common.h"
#include "flowCUDA.h"

// our project utilities
#include "IO_helper_functions.h"


///////////////////////////////////////////////////////////////////////////////
/// application entry point
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // ===============   CPU & GPU Setups   ===============
    // pick GPU
    findCudaDevice(argc, (const char **)argv);

    // remove CUDA timing latency
	cudaFree(0);
    cudaSetDevice(0);

	// read the images
	cv::Mat img_ref = cv::imread("../tests/solver_accuracy/data/img_reference.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat img_cap = cv::imread("../tests/solver_accuracy/data/img_capture.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (img_ref.empty() || img_cap.empty())
	{
		printf("Test images are not found; Program exits.\n");
		return -1;
	}

	// get sensor size
	int SENSOR_WIDTH  = img_ref.cols;
	int SENSOR_HEIGHT = img_ref.rows;

    // image dimensions
    int M_width  = 992;
    int M_height = 992;
    int M_stride = iAlignUp(M_width);
	
    // define unknown size
    int N_width  = 1024;
    int N_height = 1024;
    
    // get extra dimensions
    const static int L_width  = (N_width  - M_width) /2;
    const static int L_height = (N_height - M_height)/2;
    
    // define cropping coordinate
    const static int x_crop = (SENSOR_WIDTH  - M_width) /2;
    const static int y_crop = (SENSOR_HEIGHT - M_height)/2;

	// crop the images
    img_ref = img_ref(cv::Rect(x_crop, y_crop, M_width, M_height));
    img_cap = img_cap(cv::Rect(x_crop, y_crop, M_width, M_height));
	
	// convert to float
	img_ref.convertTo(img_ref, CV_32F);
	img_cap.convertTo(img_cap, CV_32F);
	
    // ===============   Allocate Device Pointers   ===============
	
	// allocate device pointers
    float *d_img_ref, *d_img_cap, *temp_IM, *px, *py, *temp_px, *temp_py, *temp;
	checkCudaErrors(cudaMalloc(&d_img_ref, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_img_cap, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp_IM, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&px, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&py, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp_px, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp_py, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp, M_width*M_height*sizeof(float)));
	
	// transfer data
	checkCudaErrors(cudaMemcpy(d_img_ref, img_ref.ptr(0), M_width*M_height*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_img_cap, img_cap.ptr(0), M_width*M_height*sizeof(float), cudaMemcpyHostToDevice));

    // ===============   Algorithm Setups   ===============

	// set rof parameters
	float theta = 0.125f;
	int iter = 10;
	float alp = 0.95f;

	// run FISTA ROF
	fista_rof(d_img_ref, temp_IM, px, py, temp_px, temp_py, temp, M_width, M_height, theta, iter, alp);
	fista_rof(d_img_cap, temp_IM, px, py, temp_px, temp_py, temp, M_width, M_height, theta, iter, alp);

	// save data
	float *h_test = new float [M_width*M_height];
	checkCudaErrors(cudaMemcpy(h_test, d_img_ref, M_width*M_height*sizeof(float), cudaMemcpyDeviceToHost));
	WriteFloFile("../tests/solver_accuracy/test_fista_rof.flo", M_width, M_height, h_test, h_test);

	// cleanup
	checkCudaErrors(cudaFree(d_img_ref));
	checkCudaErrors(cudaFree(d_img_cap));
	checkCudaErrors(cudaFree(temp_IM));
	checkCudaErrors(cudaFree(px));
	checkCudaErrors(cudaFree(py));
	checkCudaErrors(cudaFree(temp_px));
	checkCudaErrors(cudaFree(temp_py));
	checkCudaErrors(cudaFree(temp));
	delete[] h_test;

	return 0;
}

