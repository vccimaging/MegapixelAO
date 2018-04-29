#ifndef PREPARE_PRECOMPUTATIONS_H
#define PREPARE_PRECOMPUTATIONS_H

void prepare_precomputations(int N_W,          // unknown phase width
                             int N_H,          // unknown phase height
                             int currentLevel,
                             int *pW_N, int *pH_N, int *pS_N,
                             int *pW_M, int *pH_M, int *pS_M,
                             int *pW_L, int *pH_L,
                             float **pI0,
                             float **pI1,
                             float **d_I0_coeff,
                             float alpha,      // smoothness coefficient
                             float *mu,        // proximal parameter
                             const float **mat_x_hat, // pre-computed mat_x_hat
                             const complex **ww_1,    // ww_1 coefficient
                             const complex **ww_2);   // ww_2 coefficient
#endif
