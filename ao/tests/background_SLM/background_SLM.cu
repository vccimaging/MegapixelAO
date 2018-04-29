// Please keep this include order to make sure a successful compilation!
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <cuda_profiler_api.h> // for profiling
#include <helper_cuda.h> 

// OpenGL+GLFW
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// OpenCV
#include <opencv2/highgui.hpp>

// put CUDA-OpenGL interop head after Glad and GLFW
#include <cuda_gl_interop.h>
#include <helper_functions.h>

// our project utilities
#include "common.h"
#include "cuda_gl_interop_helper_functions.h"
#include "glfw_helper_functions.h"


__global__
void set_mod_Kernel(unsigned char *img, int width, int height)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix >= width || iy >= height)    return;

    img[ix + iy * width] = (ix + iy) % 255;
}


void set_mod_texture(unsigned char *img, int width, int height)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));

    set_mod_Kernel<<<blocks, threads>>>(img, width, height);
}



int main(int argc, char **argv)
{
    glfwSetErrorCallback(error_callback);
    
    // initialize GLFW
    if (!glfwInit())
        exit(EXIT_FAILURE);

    // detect and print monitor information
    int monitor_count, SLM_WIDTH, SLM_HEIGHT, SLM_no;
    GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
    detect_and_show_SLM(monitors, monitor_count, SLM_WIDTH, SLM_HEIGHT, SLM_no);
    if (monitor_count < 2)
    {
        printf("The SLM is not connected; Program failed.\n");
        return -1;
    }
    
    // open a window fullscreen on the SLM
    GLFWwindow* window = open_window("SLM phase (wrapped)", NULL, monitors[SLM_no], 0, 0);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glViewport(0, 0, SLM_WIDTH, SLM_HEIGHT);
    
    // set V-sync for the SLM
    glfwSwapInterval(1);
    static int swap_tear = (glfwExtensionSupported("WGL_EXT_swap_control_tear") ||
                            glfwExtensionSupported("GLX_EXT_swap_control_tear"));

    // load shaders
	GLuint program = LoadShader("../src/background_SLM.vert", 
		    					"../src/background_SLM.frag");

    // set the four coordinates for texture (the mapping)
    float x1 = -0.5f, x2 = -0.5f, x3 =  1.5f, x4 = 1.5f;
    float y1 =  1.5f, y2 = -0.5f, y3 = -0.5f, y4 = 1.5f;

    // prepare VAO
    GLuint VBO, VAO, EBO;
    prepare_VBO_VAO_EBO(VBO, VAO, EBO, x1, x2, x3, x4, y1, y2, y3, y4);

	// create texture on OpenGL
	GLuint tex_rec = create_texture(SLM_WIDTH, SLM_HEIGHT);
    GLuint tex_ref = create_texture(SLM_WIDTH, SLM_HEIGHT);

	// prepare the texture positions (please be sure to follow the *.frag) for the fragment shader
    prepare_texture_shader(program);
    
    // define a pointer to hold the SLM image on device
	unsigned char *d_SLM_img; 
	checkCudaErrors(cudaMalloc(&d_SLM_img, SLM_WIDTH*SLM_HEIGHT*sizeof(unsigned char)));
    checkCudaErrors(cudaMemset(d_SLM_img, 0, SLM_WIDTH*SLM_HEIGHT*sizeof(unsigned char)));
    
	// temporary cuda Array
    cudaArray *cuArray;

	// initialize CUDA OpenGL interop; register the resource tex_rec
    if (cuda_gl_interop_setup_texture(tex_rec))
    {
        printf("Cannot setup CUDA OpenGL interop; program failed.\n");
        return -1;
    }
    
    // initialize tex_rec with zeros
    cuda2gl(d_SLM_img, cuArray, SLM_WIDTH, SLM_HEIGHT);
    
	// initialize CUDA OpenGL interop; register the resource tex_ref
    if (cuda_gl_interop_setup_texture(tex_ref))
    {
        printf("Cannot setup CUDA OpenGL interop; program failed.\n");
        return -1;
    }

    // modify d_SLM_img
    set_mod_texture(d_SLM_img, SLM_WIDTH, SLM_HEIGHT);

    // initialize tex_ref with d_SLM_img
    cuda2gl(d_SLM_img, cuArray, SLM_WIDTH, SLM_HEIGHT);
    
    // render loop
    glfwMakeContextCurrent(window);
    while (!glfwWindowShouldClose(window))
    {
        // draw quad
        draw_quad_shader(program, VAO, tex_rec, tex_ref);
        
        glfwSwapBuffers(window);
        // glfwPollEvents();
    }

    // cleanup
	checkCudaErrors(cudaFree(d_SLM_img));
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

