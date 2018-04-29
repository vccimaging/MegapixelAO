#ifndef GLFW_HELPER_FUNCTIONS_H
#define GLFW_HELPER_FUNCTIONS_H

#include "IO_helper_functions.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>
#include <vector>

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}


static GLFWwindow* open_window(const char* title, GLFWwindow* share, GLFWmonitor* monitor, int posX, int posY)
{
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    // set window properties
    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
    
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, title, monitor, share);
    if (!window)
        return NULL;

    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

    glfwSwapInterval(1);
    glfwSetWindowPos(window, posX, posY);
    glfwShowWindow(window);

    return window;
}


// Boundary options:
// - 0: All-black boundary condition
// - 1: Replicate boundary condition
static GLuint create_texture(int width, int height, unsigned char *pixels = NULL, int boundary_opt = 0)
{
    GLuint texture;

    // create one OpenGL texture
    glGenTextures(1, &texture);
    
    // bind the texture: all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, texture);

    // initialize texture with the pixel values (with 3rd input), or zero initialized (without 3rd input)
    if (pixels == NULL){
        int x, y;
        unsigned char *pixels = new unsigned char[width * height];
	    for (y = 0; y < height; y++)
            for (x = 0; x < width; x++)
                pixels[x * height + y] = x;
    }
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, pixels);
    
    // set properties
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    
    switch(boundary_opt)
    {
        case 0:
        	printf("Texture: Zero boundary padding\n");
            // all-black boundary condition
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,  GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,  GL_CLAMP_TO_BORDER);
            break;
        case 1:
        	printf("Texture: Replicate boundary padding\n");
            // replicate boundary condition
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,  GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,  GL_CLAMP_TO_EDGE);
            break;
    };

	delete [] pixels;
    return texture;
}


GLuint LoadShader(const char *vertex_path, const char *fragment_path)
{
    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

    // Read shaders
    std::string vertShaderStr = readFile(vertex_path);
    std::string fragShaderStr = readFile(fragment_path);
    const char *vertShaderSrc = vertShaderStr.c_str();
    const char *fragShaderSrc = fragShaderStr.c_str();

    GLint result = GL_FALSE;
    int logLength;

    // Compile vertex shader
    std::cout << "Compiling vertex shader." << std::endl;
    glShaderSource(vertShader, 1, &vertShaderSrc, NULL);
    glCompileShader(vertShader);

    // Check vertex shader
    glGetShaderiv(vertShader, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> vertShaderError((logLength > 1) ? logLength : 1);
    glGetShaderInfoLog(vertShader, logLength, NULL, &vertShaderError[0]);
    std::cout << &vertShaderError[0] << std::endl;

    // Compile fragment shader
    std::cout << "Compiling fragment shader." << std::endl;
    glShaderSource(fragShader, 1, &fragShaderSrc, NULL);
    glCompileShader(fragShader);

    // Check fragment shader
    glGetShaderiv(fragShader, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> fragShaderError((logLength > 1) ? logLength : 1);
    glGetShaderInfoLog(fragShader, logLength, NULL, &fragShaderError[0]);
    std::cout << &fragShaderError[0] << std::endl;

    std::cout << "Linking program." << std::endl;
    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &result);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    std::vector<char> programError( (logLength > 1) ? logLength : 1 );
    glGetProgramInfoLog(program, logLength, NULL, &programError[0]);
    std::cout << &programError[0] << std::endl;

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return program;
}


static void prepare_VBO_VAO_EBO(GLuint& VBO, GLuint& VAO, GLuint& EBO,
                                float x1 = 0.f, float x2 = 0.f, float x3 = 1.f, float x4 = 1.f, 
                                float y1 = 1.f, float y2 = 0.f, float y3 = 0.f, float y4 = 1.f)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    float vertices[] = {
        // positions          // texture coords (recon)     // texture coords (ref)
         1.0f,  1.0f, 0.0f,   x4, y4,                       1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   x3, y3,                       1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   x2, y2,                       0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   x1, y1,                       0.0f, 1.0f  // top left 
    };
    
    unsigned int indices[] = {  // note that we start from 0!
        0, 1, 3,  // first Triangle
        1, 2, 3   // second Triangle
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute (recon)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute (ref)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0); 
}


static void prepare_texture_shader(GLuint program)
{
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "tex_recon"), 0);
    glUniform1i(glGetUniformLocation(program, "tex_ref"),   1);
}


static void prepare_VBO_VAO_EBO_simple(GLuint& VBO, GLuint& VAO, GLuint& EBO,
                                float x1 = 0.f, float x2 = 0.f, float x3 = 1.f, float x4 = 1.f, 
                                float y1 = 1.f, float y2 = 0.f, float y3 = 0.f, float y4 = 1.f)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    float vertices[] = {
        // positions          // texture coords (recon)     // texture coords (ref)
         1.0f,  1.0f, 0.0f,   x4, y4,                       1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   x3, y3,                       1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   x2, y2,                       0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   x1, y1,                       0.0f, 1.0f  // top left 
    };
    
    unsigned int indices[] = {  // note that we start from 0!
        0, 1, 3,  // first Triangle
        1, 2, 3   // second Triangle
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute (recon)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0); 
}


static void prepare_texture_shader_simple(GLuint program)
{
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "tex_recon"), 0);
}


static void draw_quad_shader(GLuint program, GLuint tex_recon, GLuint VAO, GLuint tex_ref = 0)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if (tex_ref == 0)
        glBindTexture(GL_TEXTURE_2D, tex_recon);
    else{
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_recon);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, tex_ref);
    }
    
    // draw our first triangle
    glUseProgram(program);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}


 // The four coordinates correspond to the four vertex coordates on the screen:
 // 1 4
 // 2 3
 static void draw_quad(GLuint texture, bool flipud = false,
                       float x1 = 0.f, float x2 = 0.f, float x3 = 1.f, float x4 = 1.f, 
                       float y1 = 1.f, float y2 = 0.f, float y3 = 0.f, float y4 = 1.f)
 {
     int width, height;
     glfwGetFramebufferSize(glfwGetCurrentContext(), &width, &height);
 
     glClear(GL_COLOR_BUFFER_BIT);
     
     glViewport(0, 0, width, height);
 
     glMatrixMode(GL_PROJECTION);
     glLoadIdentity();
     glOrtho(0.f, 1.f, 0.f, 1.f, 0.f, 1.f);
 
     glEnable(GL_TEXTURE_2D);
     glBindTexture(GL_TEXTURE_2D, texture);
     glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
 
     glBegin(GL_QUADS);
 
     if (flipud){ // flip upside down
         // coordate 2 -> 1
         glTexCoord2f(x1, y1);
         glVertex2f(0.f, 0.f);
 
         // coordate 3 -> 4
         glTexCoord2f(x4, y4);
         glVertex2f(1.f, 0.f);
 
         // coordate 4 -> 3
         glTexCoord2f(x3, y3);
         glVertex2f(1.f, 1.f);
 
         // coordate 1 -> 2
         glTexCoord2f(x2, y2);
         glVertex2f(0.f, 1.f);
     }
     else{ // don't flip upside down
         // coordate 2
         glTexCoord2f(x2, y2);
         glVertex2f(0.f, 0.f);
 
         // coordate 3
         glTexCoord2f(x3, y3);
         glVertex2f(1.f, 0.f);
 
         // coordate 4
         glTexCoord2f(x4, y4);
         glVertex2f(1.f, 1.f);
 
         // coordate 1
         glTexCoord2f(x1, y1);
         glVertex2f(0.f, 1.f);
     }
     
     glEnd();
 }


static void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    printf("Framebuffer resized to %ix%i\n", width, height);

    glViewport(0, 0, width, height);
}


static const char* format_mode(const GLFWvidmode* mode)
{
    static char buffer[512];

    sprintf(buffer,
            "%i x %i x %i (%i %i %i) %i Hz",
            mode->width, mode->height,
            mode->redBits + mode->greenBits + mode->blueBits,
            mode->redBits, mode->greenBits, mode->blueBits,
            mode->refreshRate);

    buffer[sizeof(buffer) - 1] = '\0';
    return buffer;
}


static void list_modes(GLFWmonitor* monitor)
{
    int count, x, y, widthMM, heightMM, i;
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    const GLFWvidmode* modes = glfwGetVideoModes(monitor, &count);

    glfwGetMonitorPos(monitor, &x, &y);
    glfwGetMonitorPhysicalSize(monitor, &widthMM, &heightMM);

    printf("Name: %s (%s)\n",
           glfwGetMonitorName(monitor),
           glfwGetPrimaryMonitor() == monitor ? "primary" : "secondary");
    printf("Current mode: %s\n", format_mode(mode));
    printf("Virtual position: %i %i\n", x, y);

    printf("Physical size: %i x %i mm (%0.2f dpi)\n",
           widthMM, heightMM, mode->width * 25.4f / widthMM);

    printf("Modes:\n");

    for (i = 0; i < count; i++)
    {
        printf("%3u: %s", (unsigned int) i, format_mode(modes + i));

        if (memcmp(mode, modes + i, sizeof(GLFWvidmode)) == 0)
            printf(" (current mode)");

        putchar('\n');
    }
}


static void detect_and_show_SLM(GLFWmonitor** monitors, int monitor_count, int& SLM_WIDTH, int& SLM_HEIGHT, int& SLM_no)
{
    printf("Number of monitors detected: %d\n", monitor_count);
    for (int i = 0; i < monitor_count; i++)
        list_modes(monitors[i]);
    if (monitor_count < 2)
        printf("The SLM is not connected.\n");
    
    SLM_no = 1; // default: SLM is the second display
    const GLFWvidmode* mode = glfwGetVideoMode(monitors[SLM_no]);
    SLM_WIDTH  = mode->width;
    SLM_HEIGHT = mode->height;
}
#endif
