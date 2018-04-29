#include "common.h"

__global__ void visualKernel(float *phi, float *in, int width, int height, int visual_opt, float scale_factor, 
							 float min_phase, float max_phase)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = ix + iy * width;

    if (ix >= width || iy >= height) return;
    
    float temp = - scale_factor * in[pos]; // telescope
    
    switch (visual_opt)
    {
        case 1: // interference rings
        {
			temp = cosf(temp/2);
            phi[pos] = temp * temp;
            break;
        }
        case 2: // height map
        {
			temp -= min_phase;
			phi[pos] = (float)((int)( 255.0f * temp / (max_phase - min_phase) )) / 255.0f;
            break;
        }
        case 3: // height map (wrapped)
        // Wrap to [-0.5 0.5], then add 0.5 to [0 1] for final phase show
        {
            phi[pos] = wrap(temp);
            break;
        }
    }
}


static
void visual(float *phi, float *in, int w, int h, int visual_opt, float scale_factor, float min_phase, float max_phase)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

    visualKernel<<<blocks, threads>>>(phi, in, w, h, visual_opt, scale_factor, min_phase, max_phase);
}
