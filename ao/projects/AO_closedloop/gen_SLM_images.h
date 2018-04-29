#include "common.h"
extern
void gen_SLM_img(unsigned char *SLM_phase, 
				 cv::cuda::GpuMat phi_gpumat, cv::cuda::GpuMat temp_gpumat, cv::cuda::GpuMat SLM_gpumat,  
				 int width, int height, int SLM_width, int SLM_height, float scale_factor, float AO_gain, int kernel_size, 
				 cv::Mat G0, float& RMS_SLM, int BOUND_COND);
extern void test_acc_SLM(float *SLM_phase, float *phi, int width, int height, float scale_factor, float AO_gain);
