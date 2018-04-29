#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief compute proximal operator of g
///
/// \param[in/out]  w_x       flow along x
/// \param[in/out]  w_y       flow along y
/// \param[in/out]  zeta_x    dual variable zeta along x
/// \param[in/out]  zeta_y    dual variable zeta along y
/// \param[in/out]  temp_x    temp variable (either for nabla(x) or w-zeta) along x
/// \param[in/out]  temp_y    temp variable (either for nabla(x) or w-zeta) along y
/// \param[in]      mu        proximal parameter
/// \param[in]      w11, w12_or_w22, w13, w21, w23   
///                           pre-computed weights
/// \param[in]      N_width   unknown width
/// \param[in]      N_height  unknown height
/// \param[in]      M_width   image width
/// \param[in]      M_height  image height
///////////////////////////////////////////////////////////////////////////////
__global__ void prox_gKernel(float *w_x, float *w_y, float *zeta_x, float *zeta_y, 
                             float *temp_x, float *temp_y, float mu,
                             const float *w11, const float *w12_or_w22, 
                             const float *w13, const float *w21, const float *w23, 
                             int N_width, int N_height, 
                             int M_width, int M_height)
{
    const int L_width  = (N_width  - M_width)/2;
    const int L_height = (N_height - M_height)/2;
    
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos_N = ix + iy * N_width;
    const int pos_M  = (ix-L_width) + (iy-L_height) * M_width;
    
    float temp_w_x = temp_x[pos_N] + zeta_x[pos_N];
    float temp_w_y = temp_y[pos_N] + zeta_y[pos_N];
    float val_x, val_y;
    
    // w-update
    if (ix >= N_width || iy >= N_height) return;
    else if (ix >= L_width  && ix < N_width -L_width && 
             iy >= L_height && iy < N_height-L_height){ // update interior flow
        val_x = w11[pos_M] * temp_w_x + w12_or_w22[pos_M] * temp_w_y + w13[pos_M];
        val_y = w21[pos_M] * temp_w_y + w12_or_w22[pos_M] * temp_w_x + w23[pos_M];
    }
    else{ // keep exterior flow unchanged
        val_x = temp_w_x;
        val_y = temp_w_y;
    }
    w_x[pos_N] = val_x;
    w_y[pos_N] = val_y;
        
    // zeta-update
    zeta_x[pos_N] = temp_w_x - val_x;
    zeta_y[pos_N] = temp_w_y - val_y;
    
    // pre-store value of (w - zeta)
    temp_x[pos_N] = mu * (2*val_x - temp_w_x);
    temp_y[pos_N] = mu * (2*val_y - temp_w_y);
}


///////////////////////////////////////////////////////////////////////////////
/// \brief compute proximal operator of g
///
/// \param[in/out]  w_x       flow along x
/// \param[in/out]  w_y       flow along y
/// \param[in/out]  zeta_x    dual variable zeta along x
/// \param[in/out]  zeta_y    dual variable zeta along y
/// \param[in/out]  temp_x    temp variable (either for nabla(x) or w-zeta) along x
/// \param[in/out]  temp_y    temp variable (either for nabla(x) or w-zeta) along y
/// \param[in]      mu        proximal parameter
/// \param[in]      w11, w12_or_w22, w13, w21, w23   
///                           pre-computed weights
/// \param[in]      N_width   unknown width
/// \param[in]      N_height  unknown height
/// \param[in]      M_width   image width
/// \param[in]      M_height  image height
///////////////////////////////////////////////////////////////////////////////
static
void prox_g(float *w_x, float *w_y, float *zeta_x, float *zeta_y, 
            float *temp_x, float *temp_y, float mu,
            const float *w11, const float *w12_or_w22, 
            const float *w13, const float *w21, const float *w23, 
            int N_width, int N_height,  int M_width, int M_height)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(N_width, threads.x), iDivUp(N_height, threads.y));

    prox_gKernel<<<blocks, threads>>>(w_x, w_y, zeta_x, zeta_y, temp_x, temp_y,
                                       mu, w11, w12_or_w22, w13, w21, w23, 
                                      N_width, N_height, M_width, M_height);
}
