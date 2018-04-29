// Please keep this include order to make sure a successful compilation!
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <cuda_profiler_api.h> // for profiling
#include <helper_cuda.h> 

// OpenGL+GLFW
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// put CUDA-OpenGL interop head after Glad and GLFW
#include <cuda_gl_interop.h>
#include <helper_functions.h>

// our project utilities
#include "cuda_gl_interop_helper_functions.h"
#include "glfw_helper_functions.h"


GLuint create_cust_texture(int width, int height)
{
    char *pixels = new char [width * height];
    GLuint texture;

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    for (int y = 0;  y < height;  y++)
        for (int x = 0;  x < width;  x++)
            pixels[y * width + x] = (x+y) % 255;

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,  GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,  GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

	delete[] pixels;

    return texture;
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
	GLuint tex_rec = create_cust_texture(SLM_WIDTH/2, SLM_HEIGHT/2);
    GLuint tex_ref = create_cust_texture(SLM_WIDTH, SLM_HEIGHT);

	// prepare the texture positions (please be sure to follow the *.frag) for the fragment shader
    prepare_texture_shader(program);
    
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
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

