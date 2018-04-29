#include "common.h"

// include kernels
#include "derivativesKernel.cuh"
#include "warpingKernel.cuh"
#include "addKernel.cuh"
#include "nablaKernel.cuh"
#include "nablaTKernel.cuh"
#include "x_updateKernel.cuh"
#include "prox_gKernel.cuh"
#include "medianfilteringKernel.cuh"

// for pyramid
extern texture<float, 2, cudaReadModeElementType> texFine;
extern
void Downscale_Anti(const float *src, int width, int height, int stride,
               int newWidth, int newHeight, int newStride, float *out);
#include "upscaleKernel.cuh"

// visualization
#include "visualize.cuh"

// FISTA ROF
#include "fista_rofKernel.cuh"

// thrust headers
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <cmath>

// thrust operators 
template <typename T>
struct absfun
{
    __device__
        T operator()(const T& x) const { 
            return fabs(x);
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

///////////////////////////////////////////////////////////////////////////////
/// \brief method logic
///
/// Our wavefront solver
/// \param[in]  pI0           reference image
/// \param[in]  pI1           captured image
/// \param[in]  pN_W          unknown phase width
/// \param[in]  pN_H          unknown phase height
/// \param[in]  pM_W          known image width
/// \param[in]  pM_H          known image height
/// \param[in]  alpha         smoothness tradeoff parameter
/// \param[in]  mu            proximal parameter
/// \param[in]  nWarpIters    number of warping iterations per pyramid level
/// \param[in]  nAlgoIters    number of proximal algorithm iterations
/// \param[in]  nLevels       number of pyramid levels
/// \param[in]  endLevel      end of pyramid levels
///----------------------------------------------------------------------------
/// \param[out] phi           final wavefront solution phi
///----------------------------------------------------------------------------
/// \param[tmp] d_I0          temporary storage for pyramid images
/// \param[tmp] w_x           ADMM slack variable w (x direction)
/// \param[tmp] w_y           ..................... (y direction)
/// \param[tmp] zeta_x        ADMM dual variable zeta (x direction)
/// \param[tmp] zeta_y        ....................... (y direction)
/// \param[tmp] ww_1          temporary storages 
/// \param[tmp] ww_2          .
/// \param[tmp] temp_x        .
/// \param[tmp] temp_y        .
/// \param[tmp] temp_DCT      .
///----------------------------------------------------------------------------
/// \param[const] mat_x_hat   DCT inversion basis
/// \param[const] w11         proximal operator constants
/// \param[const] w13         .
/// \param[const] w12_or_w22  .
/// \param[const] w21         .
/// \param[const] w23         .
///----------------------------------------------------------------------------
/// \param[doc] RMS_phi       phase RMS
/// \param[doc] nonzero_phase number of nonzero elements
/// \param[doc] phi_center    center of solution
/// \param[doc] h_test        host test variable
/// \param[doc] scale_factor  phase scale factor
///////////////////////////////////////////////////////////////////////////////
void ComputeFlowCUDA(float **pI0, float **pI1,
                     int N_W, int N_H,
                     int M_W, int M_H,
                     int *pW_N, int *pH_N, int *pS_N, 
                     int *pW_M, int *pH_M, int *pS_M, 
                     float alpha, float *mu,
                     int *nWarpIters,
                     int nAlgoIters,
                     int nLevels, int endLevel,
                     cufftHandle plan_dct_1,
                     cufftHandle plan_dct_2,
                     cufftHandle plan_dct_3,
                     cufftHandle plan_dct_4,
                     cufftHandle plan_dct_5,
                     cufftHandle plan_dct_6,
                     cufftHandle plan_dct_7,
                     cufftHandle plan_dct_8,
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
                     float scale_factor = 0.05)
{
//    printf("Solving first-order DCT optimization problem using ADMM on GPU...\n");

    // define reduction pointer
    thrust::device_ptr<float> d_ptr;
    
    // thrust setup arguments
    absfun<float>       unary_op;
    absfun2<float>      unary_op2;
    thrust::plus<float> binary_op;

	// get the energy of the measured image image_cap
    d_ptr = thrust::device_pointer_cast(pI0[nLevels-1]);
    float energy_I0 = thrust::transform_reduce(d_ptr, d_ptr + M_W*M_H, unary_op, 0.0f, binary_op);
    d_ptr = thrust::device_pointer_cast(pI1[nLevels-1]);
    float energy_I1 = thrust::transform_reduce(d_ptr, d_ptr + M_W*M_H, unary_op, 0.0f, binary_op);
	if ( (energy_I1 < 0.6*energy_I0) || (energy_I1 > 1.6*energy_I0) )
	{
		printf("Image energy mismatch; Solver refuses to work; Set zero phase as ouput ...\n");
        checkCudaErrors(cudaMemset(phi[nLevels-1], 0, N_W*N_H*sizeof(float)));
		return;
	}
	
    int currentLevel = nLevels - 1;

    // get pyramid
    for (; currentLevel > endLevel; currentLevel--)
        Downscale_Anti(pI1[currentLevel], pW_M[currentLevel], pH_M[currentLevel], pS_M[currentLevel], 
                  	  pW_M[currentLevel-1], pH_M[currentLevel-1], pS_M[currentLevel-1],
                  	  (float *)pI1[currentLevel-1]);
    
    // initialize
    int dataSize_N = pW_N[currentLevel]*pH_N[currentLevel]*sizeof(float);
	checkCudaErrors(cudaMemset(phi[currentLevel],    0, dataSize_N));
    checkCudaErrors(cudaMemset(temp_x[currentLevel], 0, dataSize_N));
    checkCudaErrors(cudaMemset(temp_y[currentLevel], 0, dataSize_N));
    
    // do it in the pyramid (pyramid warping)
	for (; currentLevel < nLevels; currentLevel++)
	{
//        printf("currentLevel = %d ... \n", currentLevel);
//	    printf("N width = %d, N height = %d \n", pW_N[currentLevel], pH_N[currentLevel]);
//	    printf("M width = %d, M height = %d \n", pW_M[currentLevel], pH_M[currentLevel]);
	
        dataSize_N = pW_N[currentLevel]*pH_N[currentLevel]*sizeof(float);
    
        // in-level warping
	    for (int warpIter = 0; warpIter < nWarpIters[currentLevel]; warpIter++){
	
//			printf("wrap: %d\n", warpIter);
	
            // calculate flow
	        nabla(phi[currentLevel], pW_N[currentLevel], pH_N[currentLevel],
	              temp_x[currentLevel], temp_y[currentLevel]);

	        // warp the image
            if (!(currentLevel == 0 && warpIter == 0))
	            WarpImage(d_I0[currentLevel], pW_N[currentLevel], pH_N[currentLevel],
	                      pW_M[currentLevel], pH_M[currentLevel], pS_M[currentLevel],
                          temp_x[currentLevel], temp_y[currentLevel], 
                          pI0[currentLevel]); // this step we need the center only
            
	        // pre-compute prox_g weights and store them in device
	        ComputeDerivatives(pI0[currentLevel], pI1[currentLevel], 
	                           pW_M[currentLevel], pH_M[currentLevel], pS_M[currentLevel],
	                           mu[currentLevel], 
	                           w11[currentLevel], w12_or_w22[currentLevel], w13[currentLevel], 
	                           w21[currentLevel], w23[currentLevel]);
	                           
            // initialize slack variable w, and dual variable zeta
            checkCudaErrors(cudaMemset(phi_delta[currentLevel], 0, dataSize_N));
            checkCudaErrors(cudaMemset(w_x[currentLevel],       0, dataSize_N));
            checkCudaErrors(cudaMemset(w_y[currentLevel],       0, dataSize_N));
            checkCudaErrors(cudaMemset(zeta_x[currentLevel],    0, dataSize_N));
            checkCudaErrors(cudaMemset(zeta_y[currentLevel],    0, dataSize_N));
            
            // do the proximal algorithm: we are using ADMM here
            for (int k = 0; k < nAlgoIters; k++){
            
                // 1. phi-update step
                if (k != 0){
                    nablaT(temp_x[currentLevel], temp_y[currentLevel], 
                           pW_N[currentLevel], pH_N[currentLevel], phi_delta[currentLevel]);
                    switch (currentLevel){
                        case 0:
                            x_update(phi_delta[currentLevel], temp_dct[currentLevel], mat_x_hat[currentLevel], 
                                     ww_1[currentLevel], ww_2[currentLevel], 
                                     pW_N[currentLevel], pH_N[currentLevel], plan_dct_1, plan_dct_2);
                            break;
                        case 1:
                            x_update(phi_delta[currentLevel], temp_dct[currentLevel], mat_x_hat[currentLevel], 
                                     ww_1[currentLevel], ww_2[currentLevel], 
                                     pW_N[currentLevel], pH_N[currentLevel], plan_dct_3, plan_dct_4);
                            break;
                        case 2:
                            x_update(phi_delta[currentLevel], temp_dct[currentLevel], mat_x_hat[currentLevel], 
                                     ww_1[currentLevel], ww_2[currentLevel], 
                                     pW_N[currentLevel], pH_N[currentLevel], plan_dct_5, plan_dct_6);
                            break;
                        case 3:
                            x_update(phi_delta[currentLevel], temp_dct[currentLevel], mat_x_hat[currentLevel], 
                                     ww_1[currentLevel], ww_2[currentLevel], 
                                     pW_N[currentLevel], pH_N[currentLevel], plan_dct_7, plan_dct_8);
                            break;
                    }
                }

                // pre-compute nabla(phi) and store it in temp
                nabla(phi_delta[currentLevel], pW_N[currentLevel], pH_N[currentLevel],
                      temp_x[currentLevel], temp_y[currentLevel]);
            
                // 2. w-update step & 3. zeta-update step
                // we pre-store (w-zeta) in temp
                prox_g(w_x[currentLevel], w_y[currentLevel], 
                       zeta_x[currentLevel], zeta_y[currentLevel], 
                       temp_x[currentLevel], temp_y[currentLevel], 
                       mu[currentLevel], 
                       w11[currentLevel], w12_or_w22[currentLevel], w13[currentLevel], 
                       w21[currentLevel], w23[currentLevel], 
                       pW_N[currentLevel], pH_N[currentLevel],
                       pW_M[currentLevel], pH_M[currentLevel]);
            }
            
            // do median filtering on the estimated gradient to suppress noise (zeta_x and zeta_y are for temporary storage)
            median2(temp_x[currentLevel], pW_N[currentLevel], pH_N[currentLevel], zeta_x[currentLevel]);
            median2(temp_y[currentLevel], pW_N[currentLevel], pH_N[currentLevel], zeta_y[currentLevel]);

            // estimate the final phase
            nablaT(zeta_x[currentLevel], zeta_y[currentLevel], 
                   pW_N[currentLevel], pH_N[currentLevel], phi_delta[currentLevel]);
                   
            switch (currentLevel){
                case 0:
                    x_update(phi_delta[currentLevel], temp_dct[currentLevel], mat_x_hat[currentLevel], 
                             ww_1[currentLevel], ww_2[currentLevel], 
                             pW_N[currentLevel], pH_N[currentLevel], plan_dct_1, plan_dct_2);
                    break;
                case 1:
                    x_update(phi_delta[currentLevel], temp_dct[currentLevel], mat_x_hat[currentLevel], 
                             ww_1[currentLevel], ww_2[currentLevel], 
                             pW_N[currentLevel], pH_N[currentLevel], plan_dct_3, plan_dct_4);
                    break;
                case 2:
                    x_update(phi_delta[currentLevel], temp_dct[currentLevel], mat_x_hat[currentLevel], 
                             ww_1[currentLevel], ww_2[currentLevel], 
                             pW_N[currentLevel], pH_N[currentLevel], plan_dct_5, plan_dct_6);
                    break;
                case 3:
                    x_update(phi_delta[currentLevel], temp_dct[currentLevel], mat_x_hat[currentLevel], 
                             ww_1[currentLevel], ww_2[currentLevel], 
                             pW_N[currentLevel], pH_N[currentLevel], plan_dct_7, plan_dct_8);
                    break;
            }
            
            // calculate the mean of delta phi
            if (currentLevel == 0 && warpIter == 0)
                d_ptr = thrust::device_pointer_cast(phi[currentLevel]);
            else
                d_ptr = thrust::device_pointer_cast(phi_delta[currentLevel]);
            float mean_phi_delta = thrust::transform_reduce(d_ptr, d_ptr + pW_N[currentLevel]*pH_N[currentLevel], 
                                                            unary_op, 0.0f, binary_op);
            mean_phi_delta /= pW_N[currentLevel]*pH_N[currentLevel];
            
            // check if the mean of phi is too small; for early termination
            if (mean_phi_delta < 0 / std::pow(4, nLevels-1 - currentLevel)) // we use mod 2 here
            {                 //[0.314 0.628 1.257 2.094] = 2*pi./[20 10 5 3]
//            	printf("Pyr %d, Warp %d, Mean of delta phi = %f < eps: Early termination \n", 
//                                          currentLevel, warpIter+1, mean_phi_delta);
                checkCudaErrors(cudaMemset(phi_delta[currentLevel], 0, dataSize_N));
//                break;
            }
//            else printf("Pyr %d, Warp %d, Mean of delta phi = %f \n", currentLevel, warpIter+1, mean_phi_delta);

            // incrementally add phi
            if (currentLevel == 0 && warpIter == 0)
                Swap(phi[currentLevel], phi_delta[currentLevel]);
            else
                Add(phi[currentLevel], phi_delta[currentLevel], 
                    pW_N[currentLevel]*pH_N[currentLevel], phi[currentLevel]);
        }
        
        // prolongate solution
        if (currentLevel != nLevels - 1){
            float scale2 = (float)pW_N[currentLevel + 1]/(float)pW_N[currentLevel] *
                           (float)pH_N[currentLevel + 1]/(float)pH_N[currentLevel];
//            printf("scale2 = %f \n", scale2);
            Upscale(phi[currentLevel], pW_N[currentLevel], pH_N[currentLevel],
                    pW_N[currentLevel + 1], pH_N[currentLevel + 1], 
                    scale2, phi[currentLevel + 1]);
        }
    }
    
    // calculate final RMS
    d_ptr = thrust::device_pointer_cast(phi[nLevels-1]);
    RMS_phi = thrust::transform_reduce(d_ptr, d_ptr + pW_N[nLevels-1]*pH_N[nLevels-1], unary_op2, 0.0f, binary_op);
    RMS_phi = sqrtf( RMS_phi*scale_factor*scale_factor / (float)(nonzero_phase) );
}



void visualize_phase(float *phi, float *in, int width, int height, 
                     int visual_opt, float scale_factor, int min_phase, int max_phase)
{
    // for visualization (note: phi is already zero-mean)
    visual(phi, in, width, height, visual_opt, scale_factor, min_phase, max_phase);
}



void scale_image(float *img, float *IMG, int width, int height, float vlow, float vhigh, 
																float ilow = 0.0f, float ihigh = 255.0f)
{
	// configure the Kernels
	dim3 threads(32, 32);
    dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
    
	// define reduction pointer
    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(img);
    
    // get the lowest value
    if (ilow == 0.0f)
    	ilow = *(thrust::min_element(d_ptr, d_ptr + width*height));

	// usually the maximum is 255; we don't need to compute this; save some time :/	
    if (ihigh == 255.0f)
    	ihigh = *(thrust::max_element(d_ptr, d_ptr + width*height));
    	
	// do the scaling
    ScaleImageKernel<<<blocks, threads>>>(img, IMG, width, height, vlow, vhigh, ilow, ihigh);
}



void fista_rof(float *im, float *IM, float *p_x, float *p_y, float *temp_p_x, float *temp_p_y, 
			   float *temp, int width, int height, float theta, int iter, float alp)
{
    // thrust setup arguments
    absfun2<float>      unary_op2;
    thrust::plus<float> binary_op;
    
	// configure the Kernels
	dim3 threads(32, 32);
    dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

	// scale the images
	scale_image(im, IM, width, height, -1.0f, 1.0f);
	checkCudaErrors(cudaMemcpy(im, IM, width*height*sizeof(float), cudaMemcpyDeviceToDevice));
	
	// set step size
	float tau = 0.25f;
	
	// initialize p
	checkCudaErrors(cudaMemset(p_x, 0, width*height*sizeof(float)));
    checkCudaErrors(cudaMemset(p_y, 0, width*height*sizeof(float)));
	
	// do the dual proximal gradient descent
	for (int k = 0; k < iter; k++)
	{
		// iteration
		nablaT(p_x, p_y, width, height, temp);
		fista_iter<<<blocks, threads>>>(temp, width, height, IM, theta);

		// (record objective)
		thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(temp);
    	float obj = thrust::transform_reduce(d_ptr, d_ptr + width*height, unary_op2, 0.0f, binary_op);
//		printf("iter = %d, obj = %.6e\n", k+1, obj);
		
		// keep on iterating
		nabla(temp, width, height, temp_p_x, temp_p_y);
		prox_LinfKernel<<<blocks, threads>>>(p_x, temp_p_x, width, height, theta, tau);
		prox_LinfKernel<<<blocks, threads>>>(p_y, temp_p_y, width, height, theta, tau);
	}

	// get the primal solution
	nablaT(p_x, p_y, width, height, temp);
	primalKernel<<<blocks, threads>>>(im, IM, width, height, theta, alp, temp);

	// scale back
	scale_image(im, im, width, height, 0.0f, 255.0f);
}
