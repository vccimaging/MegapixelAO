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
#include "prepare_cufft_warmup.h"
#include "prepare_precomputations.h"

// our project utilities
#include <IO_helper_functions.h>

// define constants
#define nLevels 1    // number of pyramid levels 

///////////////////////////////////////////////////////////////////////////////
/// application entry point
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // compute scaling factor
    float scale_factor = 0.05f; // estimated scale factor

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

    // print the cropping area origin
    printf("x: %d, y: %d \n", x_crop, y_crop);

	// crop the images
    img_ref = img_ref(cv::Rect(x_crop, y_crop, M_width, M_height));
    img_cap = img_cap(cv::Rect(x_crop, y_crop, M_width, M_height));
	
	// convert to float
	img_ref.convertTo(img_ref, CV_32F);
	img_cap.convertTo(img_cap, CV_32F);

    // ===============   Algorithm Setups   ===============

    // smoothness
    const float alpha = 3.0f;

    // number of warping iterations (0 to nLevels-1: from coarse to fine)
    int *nWarpIters = new int [nLevels];
    nWarpIters[0] = 1;

    // proximal parameter (changable if using half-quadratic-splitting)
    float *mu = new float [nLevels];
    mu[0] = 100.0f;

    // number of proximal algorithm iterations
    const int nAlgoIters = 10;

    // define variables for proximal algorithm
    float **pI0 = new float *[nLevels];
    float **pI1 = new float *[nLevels]; // store the image pyramid
    const float **mat_x_hat = new const float *[nLevels];
    float **d_I0_coeff = new float *[nLevels]; // store the cubic coefficients image pyramid
    
    int *pW_N = new int [nLevels];
    int *pH_N = new int [nLevels];
    int *pS_N = new int [nLevels];
    int *pW_M = new int [nLevels];
    int *pH_M = new int [nLevels];
    int *pS_M = new int [nLevels];
    int *pW_L = new int [nLevels];
    int *pH_L = new int [nLevels];
    
    float **phi = new float *[nLevels];
    float **phi_delta = new float *[nLevels];
    float **w_x = new float *[nLevels];
    float **w_y = new float *[nLevels];
    float **zeta_x = new float *[nLevels];
    float **zeta_y = new float *[nLevels];

    float **w11 = new float *[nLevels];
    float **w12_or_w22 = new float *[nLevels];
    float **w13 = new float *[nLevels];
    float **w21 = new float *[nLevels];
    float **w23 = new float *[nLevels];
    
    const complex **ww_1 = new const complex *[nLevels];
    const complex **ww_2 = new const complex *[nLevels];
    
    float **temp_x = new float *[nLevels];
    float **temp_y = new float *[nLevels];
    complex **temp_dct = new complex *[nLevels];
    
    int dataSize_M = M_width*M_height*sizeof(float);
    
    // prepare pyramid
    checkCudaErrors(cudaMalloc(pI0 + nLevels-1, dataSize_M));
    checkCudaErrors(cudaMalloc(pI1 + nLevels-1, dataSize_M));
    checkCudaErrors(cudaMalloc(d_I0_coeff + nLevels-1, dataSize_M));

    // pass calibration data in the device poiners
    checkCudaErrors(cudaMemcpy((void *)pI0[nLevels-1], img_ref.ptr<float>(0), dataSize_M, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *)d_I0_coeff[nLevels-1], img_ref.ptr<float>(0), dataSize_M, cudaMemcpyHostToDevice));

	// ============ do ROF denoising on the images ============
	// allocate device pointers
    float *temp_IM, *px, *py, *temp_px, *temp_py, *temp;
	checkCudaErrors(cudaMalloc(&temp_IM, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&px, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&py, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp_px, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp_py, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp, M_width*M_height*sizeof(float)));
	
	// set ROF denoising parameters
	float theta = 0.125f;
	int iter = 20;
	float alp = 0.95f;

	// run FISTA ROF on reference image
	fista_rof(pI0[nLevels-1], temp_IM, px, py, temp_px, temp_py, temp, M_width, M_height, theta, iter, alp);
	fista_rof(d_I0_coeff[nLevels-1], temp_IM, px, py, temp_px, temp_py, temp, M_width, M_height, theta, iter, alp);
	
	// =========================================================

    // determine L value for best performance (we use mod 2 here)
    for (int i = nLevels-1; i >= 0; i--){
        pW_L[i] = L_width  >> (nLevels-1 - i);
        pH_L[i] = L_height >> (nLevels-1 - i);
        if (pW_L[i] < 2)    pW_L[i] = 2;
        if (pH_L[i] < 2)    pH_L[i] = 2;
    }
    
    pW_N[nLevels-1] = M_width  + 2*pW_L[nLevels-1];
    pH_N[nLevels-1] = M_height + 2*pH_L[nLevels-1];
    pS_N[nLevels-1] = iAlignUp(pW_N[nLevels-1]);
    pW_M[nLevels-1] = M_width;
    pH_M[nLevels-1] = M_height;
    pS_M[nLevels-1] = M_stride;

    printf("initial sizes: W_N = %d, H_N = %d, S_N = %d\n", N_width, N_height, iAlignUp(pW_N[nLevels-1]));
    printf("initial sizes: W_M = %d, H_M = %d, S_M = %d\n", M_width, M_height, M_stride);
    
	// pre-computations
    prepare_precomputations(N_width, N_height, nLevels-1, 
                            pW_N, pH_N, pS_N, 
                            pW_M, pH_M, pS_M, 
                            pW_L, pH_L, pI0, pI1, d_I0_coeff, alpha, mu, 
                            mat_x_hat, ww_1, ww_2);
                       
    // allocate memory for the variables
    printf("\nAllocating device variables ...  ");
    int dataSize_N;
    for (int i = nLevels-1; i >= 0; i--){
    
        dataSize_N = pW_N[i] * pH_N[i] * sizeof(float);
        dataSize_M = pW_M[i] * pH_M[i] * sizeof(float);
            
        checkCudaErrors(cudaMalloc(phi + i, dataSize_N));
        
        checkCudaErrors(cudaMalloc(phi_delta + i, dataSize_N));
        checkCudaErrors(cudaMalloc(w_x + i, dataSize_N));
        checkCudaErrors(cudaMalloc(w_y + i, dataSize_N));
        checkCudaErrors(cudaMalloc(zeta_x + i, dataSize_N));   
        checkCudaErrors(cudaMalloc(zeta_y + i, dataSize_N));

        checkCudaErrors(cudaMalloc(w11 + i, dataSize_M));
        checkCudaErrors(cudaMalloc(w13 + i, dataSize_M));
        checkCudaErrors(cudaMalloc(w12_or_w22 + i, dataSize_M));
        checkCudaErrors(cudaMalloc(w21 + i, dataSize_M));
        checkCudaErrors(cudaMalloc(w23 + i, dataSize_M));
                        
        checkCudaErrors(cudaMalloc(temp_x + i, dataSize_N));
        checkCudaErrors(cudaMalloc(temp_y + i, dataSize_N));
        checkCudaErrors(cudaMalloc(temp_dct + i, 
                        pW_N[i]*pH_N[i]*sizeof(complex)));
    }
    printf("Done.\n");

	// prepare cufft plans & warmup
    printf("Preparing CuFFT plans and warmups ...  ");
	cufftHandle plan_dct_1, plan_dct_2,
	            plan_dct_3, plan_dct_4,
	            plan_dct_5, plan_dct_6,
	            plan_dct_7, plan_dct_8;
	int Length1[1], Length2[1];
	Length1[0] = pH_N[0]; // for each FFT, the Length1 is N_height
	Length2[0] = pW_N[0];  // for each FFT, the Length2 is N_width
	cufftPlanMany(&plan_dct_1, 1, Length1, 
					  Length1, pW_N[0], 1, 
					  Length1, pW_N[0], 1, 
					CUFFT_C2C, pW_N[0]);
	cufftPlanMany(&plan_dct_2, 1, Length2, 
					  Length2, pH_N[0], 1, 
					  Length2, pH_N[0], 1, 
					CUFFT_C2C, pH_N[0]);
	Length1[0] = pH_N[1]; // for each FFT, the Length1 is N_height
	Length2[0] = pW_N[1];  // for each FFT, the Length2 is N_width
	cufftPlanMany(&plan_dct_3, 1, Length1, 
					  Length1, pW_N[1], 1, 
					  Length1, pW_N[1], 1, 
					CUFFT_C2C, pW_N[1]);
	cufftPlanMany(&plan_dct_4, 1, Length2,
					  Length2, pH_N[1], 1, 
					  Length2, pH_N[1], 1, 
					CUFFT_C2C, pH_N[1]);
	Length1[0] = pH_N[2]; // for each FFT, the Length1 is N_height
	Length2[0] = pW_N[2];  // for each FFT, the Length2 is N_width				
	cufftPlanMany(&plan_dct_5, 1, Length1, 
					  Length1, pW_N[2], 1, 
					  Length1, pW_N[2], 1, 
					CUFFT_C2C, pW_N[2]);
	cufftPlanMany(&plan_dct_6, 1, Length2, 
					  Length2, pH_N[2], 1, 
					  Length2, pH_N[2], 1, 
					CUFFT_C2C, pH_N[2]);
	Length1[0] = pH_N[3]; // for each FFT, the Length1 is N_height
	Length2[0] = pW_N[3];  // for each FFT, the Length2 is N_width			
	cufftPlanMany(&plan_dct_7, 1, Length2, 
					  Length1, pW_N[3], 1, 
					  Length1, pW_N[3], 1, 
					CUFFT_C2C, pW_N[3]);
	cufftPlanMany(&plan_dct_8, 1, Length2, 
					  Length2, pH_N[3], 1, 
					  Length2, pH_N[3], 1, 
					CUFFT_C2C, pH_N[3]);
					
	// cufft warmup
	complex *h_warmup_in  = new complex [N_width * N_height];
	complex *h_warmup_out = new complex [N_width * N_height];
	cufft_warper(h_warmup_in, N_width, N_height, plan_dct_1, h_warmup_out);
	cufft_warper(h_warmup_in, N_width, N_height, plan_dct_2, h_warmup_out);
	delete [] h_warmup_in;
	delete [] h_warmup_out;
    printf("Done.\n");

	// define test variable
	int test_width  = N_width;
	int test_height = N_height;
	float *h_test = new float [test_width*test_height];
	float RMS = 0.0f;

    // copy data from host to device and prepare the pyramid
    checkCudaErrors(cudaMemcpy((void *)pI1[nLevels-1], img_cap.ptr<float>(0), M_width*M_height*sizeof(float), cudaMemcpyHostToDevice));

	// run FISTA ROF on captured image
	fista_rof(pI1[nLevels-1], temp_IM, px, py, temp_px, temp_py, temp, M_width, M_height, theta, iter, alp);

	// run the algorithm
	printf("Running the algorithm ...  ");
	ComputeFlowCUDA(pI0, pI1, N_width, N_height, 
                    M_width, M_height,
                    pW_N, pH_N, pS_N,
                    pW_M, pH_M, pS_M,
                    alpha, mu,
                    nWarpIters, nAlgoIters, nLevels, 0,
                    plan_dct_1, plan_dct_2,
                    plan_dct_3, plan_dct_4,
                    plan_dct_5, plan_dct_6,
                    plan_dct_7, plan_dct_8,
                    d_I0_coeff,
                    phi, phi_delta, w_x, w_y,
                    zeta_x, zeta_y, mat_x_hat,
                    w11, w13, w12_or_w22, 
                    w21, w23, ww_1, ww_2,
                    temp_x, temp_y, temp_dct, RMS, N_width*N_height,
                    NULL,
                    h_test,
                    scale_factor);
    printf("Done.\n");

	// write to flow file
	printf("Writing to disk ...  ");
	checkCudaErrors(cudaMemcpy(h_test, phi[nLevels-1], test_width*test_height*sizeof(float), cudaMemcpyDeviceToHost));
	WriteFloFile("../tests/solver_accuracy/test_solver.flo", test_width, test_height, h_test, h_test);
    printf("Done.\n");

	printf("RMS = %f lambda.\n", RMS);

	// cleanup
	delete[] h_test;
    for (int i = 0; i < nLevels; i++)
    {
        checkCudaErrors(cudaFree((void *)pI0[i]));
        checkCudaErrors(cudaFree((void *)pI1[i]));
        checkCudaErrors(cudaFree((void *)d_I0_coeff[i]));
        
        checkCudaErrors(cudaFree((void *)mat_x_hat[i]));
        checkCudaErrors(cudaFree((void *)ww_1[i]));
        checkCudaErrors(cudaFree((void *)ww_2[i]));
        
        checkCudaErrors(cudaFree((void *)phi[i]));
        checkCudaErrors(cudaFree((void *)phi_delta[i]));
        checkCudaErrors(cudaFree((void *)w_x[i]));
        checkCudaErrors(cudaFree((void *)w_y[i]));
        checkCudaErrors(cudaFree((void *)zeta_x[i]));
        checkCudaErrors(cudaFree((void *)zeta_y[i]));
        
        checkCudaErrors(cudaFree((void *)w11[i]));
        checkCudaErrors(cudaFree((void *)w13[i]));
        checkCudaErrors(cudaFree((void *)w12_or_w22[i]));
        checkCudaErrors(cudaFree((void *)w21[i]));
        checkCudaErrors(cudaFree((void *)w23[i]));
        
        checkCudaErrors(cudaFree((void *)temp_x[i]));
        checkCudaErrors(cudaFree((void *)temp_y[i]));
        checkCudaErrors(cudaFree((void *)temp_dct[i]));
    }

    delete[] pI0;
    delete[] pI1;
    delete[] mat_x_hat;
    delete[] d_I0_coeff;
    
    delete[] ww_1;
    delete[] ww_2;
    delete[] pW_N;
    delete[] pH_N;
    delete[] pS_N;
    delete[] pW_M;
    delete[] pH_M;
    delete[] pS_M;
    delete[] pW_L;
    delete[] pH_L;
    
    delete[] nWarpIters;
    delete[] mu;
    
    delete[] phi;
    delete[] phi_delta;
    delete[] w_x;
    delete[] w_y;
    delete[] zeta_x;
    delete[] zeta_y;
    
    delete[] w11;
    delete[] w13;
    delete[] w12_or_w22;
    delete[] w21;
    delete[] w23;
    
    delete[] temp_x;
    delete[] temp_y;
    delete[] temp_dct;
    
    // destroy
	cufftDestroy(plan_dct_1);
	cufftDestroy(plan_dct_2);
	cufftDestroy(plan_dct_3);
	cufftDestroy(plan_dct_4);
	cufftDestroy(plan_dct_5);
	cufftDestroy(plan_dct_6);
	cufftDestroy(plan_dct_7);
	cufftDestroy(plan_dct_8);
	
	// clean ROF denoising pointers
	checkCudaErrors(cudaFree(temp_IM));
	checkCudaErrors(cudaFree(px));
	checkCudaErrors(cudaFree(py));
	checkCudaErrors(cudaFree(temp_px));
	checkCudaErrors(cudaFree(temp_py));
	checkCudaErrors(cudaFree(temp));
	
	cudaDeviceReset();

	return 0;
}

