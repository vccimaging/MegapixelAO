#include <cufft.h>

#ifndef FLOW_CUDA_H
#define FLOW_CUDA_H

void ComputeFlowCUDA(float **pI0, float **pI1,
                     int N_W,          // unknown phase width
                     int N_H,          // unknown phase height
                     int M_W,          // known image width
                     int M_H,          // known image height
                     int *pW_N, int *pH_N, int *pS_N, 
                     int *pW_M, int *pH_M, int *pS_M, 
                     float alpha,      // smoothness coefficient
                     float *mu,        // proximal parameter
                     int *nWarpIters,  // number of warping iterations per pyramid level
                     int nAlgoIters,   // number of proximal algorithm iterations
                     int nLevels, int endLevel,
                     cufftHandle plan_dct_1, // cufft handle
                     cufftHandle plan_dct_2, // cufft handle
                     cufftHandle plan_dct_3, // cufft handle
                     cufftHandle plan_dct_4, // cufft handle
                     cufftHandle plan_dct_5, // cufft handle
                     cufftHandle plan_dct_6, // cufft handle
                     cufftHandle plan_dct_7, // cufft handle
                     cufftHandle plan_dct_8, // cufft handle
                     float **d_I0,
                     float **phi, 
                     float **phi_delta,
                     float **w_x, float **w_y,
                     float **zeta_x, float **zeta_y, const float **mat_x_hat,
                     float **w11, float **w13, float **w12_or_w22, 
                     float **w21, float **w23, const complex **ww_1, const complex **ww_2,
                     float **temp_x, float **temp_y, complex **temp_dct, float& RMS_phi, int nonzero_phase,
                     float *phi_center,
                     float *h_test,
                     float scale_factor = 0.05);    // final wavefront solution phi
					 
void visualize_phase(float *phi, float *in, int width, int height, 
                     int visual_opt, float scale_factor, int min_phase, int max_phase);
                     
void fista_rof(float *im, float *IM, float *p_x, float *p_y, float *temp_p_x, float *temp_p_y,
			   float *temp, int width, int height, float theta, int iter, float alp);
			   
void scale_image(float *img, float *IMG, int width, int height, 
				 float vlow, float vhigh, float ilow = 0.0f, float ihigh = 255.0f);
				 
void test_warp(float *phi, int N_w, int N_h, int M_w, int M_h, int M_s,
               float *u, float *v, float *d_I0, float *pI0);
#endif
