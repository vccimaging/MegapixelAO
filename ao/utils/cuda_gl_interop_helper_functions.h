#ifndef CUDA_GL_INTEROP_HELPER_FUNCTIONS_H
#define CUDA_GL_INTEROP_HELPER_FUNCTIONS_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

static cudaGraphicsResource_t cudaResource;

static bool cuda_gl_interop_setup_texture(GLuint &texture)
{
    // initialize texture
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    // register the testure
    cudaError_t  err = cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess)
    {
        std::cout << "cudaGraphicsGLRegisterImage: " << err << "Line: " << __LINE__;
        return true;
    }
    
    return false;
}

static void cuda2gl(unsigned char *p_data, cudaArray *cuArray, int width, int height)
{
    // get mapped resource as cuArray
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaResource, 0));	
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0));

    // copy data to cuArray: make modification on texture memory via CUDA
    cudaMemcpyToArray(cuArray, 0, 0, p_data, width*height*sizeof(unsigned char), cudaMemcpyDeviceToDevice);

    // ummap resource and OpenGL can now perform on texture
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
}
#endif

