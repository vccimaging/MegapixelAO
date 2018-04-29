#if defined(LINUX32) || defined(LINUX64)
#define LINUX
#endif

// standard libraries
#include <stdlib.h>
#include <iostream>

// Please keep this include order to make sure a successful compilation!
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <cuda_profiler_api.h> // for profiling
#include <helper_cuda.h> 

// flycapture SDK
#include <FlyCapture2.h>

// OpenCV
#include <opencv2/opencv.hpp>

// for software trigger
#ifdef LINUX
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#endif

// glfw
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// put CUDA-OpenGL interop and helper_functions after glad and glfw
#include <cuda_gl_interop.h>

// use software trigger for the camera sync
// (uncomment to enable hardware trigger)
#define SOFTWARE_TRIGGER_CAMERA

// our project utilities
#include "cuda_gl_interop_helper_functions.h"
#include "glfw_helper_functions.h"
#include "flycapture2_helper_functions.h"

using namespace FlyCapture2;

static GLFWwindow* windows[1];

static int swap_interval = 1;

/////////////////////////////////////////////////////////////////////////////////
///// application entry point
/////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    // ===============   GLFW Settings   ===============
    glfwSetErrorCallback(error_callback);

    // initialize GLFW
    if (!glfwInit())
        exit(EXIT_FAILURE);

    // detect and print monitor information
    int monitor_count, SLM_WIDTH, SLM_HEIGHT, SLM_no;
    GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
    detect_and_show_SLM(monitors, monitor_count, SLM_WIDTH, SLM_HEIGHT, SLM_no);
    if (monitor_count < 2){
        printf("The SLM is not connected; Program failed.\n");
        return -1;
    }

    // remove CUDA timing latency
	cudaFree(0);
    cudaSetDevice(0);  // use GeForce GTX TITAN
    
    // open a window fullscreen on the SLM
    windows[0] = open_window("SLM phase", NULL, monitors[SLM_no], 0, 0);
    if (!windows[0])
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    GLuint texture = create_texture(SLM_WIDTH, SLM_HEIGHT);

    glfwMakeContextCurrent(windows[0]);

    // ===============   GLFW Settings (END)   ===============

    // allocate host memory
    cv::Mat h_SLM_img(cv::Size(SLM_WIDTH, SLM_HEIGHT), CV_8UC1, cv::Scalar(0));
    
    // define device variables
    unsigned char *d_SLM_img;
    int dataSize_int = SLM_WIDTH*SLM_HEIGHT*sizeof(unsigned char);
    checkCudaErrors(cudaMalloc(&d_SLM_img, dataSize_int));
    checkCudaErrors(cudaMemset(d_SLM_img, 0, dataSize_int));
    
    // define variables for the camera
    Image rawImage;
    
    // for image savings
    char image_name[80];

    // ===============   Camera Settings   ===============
    Error error;
    Camera camera;                      // camera
    CameraInfo camInfo;                 // camera information
    EmbeddedImageInfo embeddedInfo;     // embedded settings
    TriggerMode triggerMode;			// camera trigger mode
    
    // initialize PointGrey
    if (!pointgrey_initialize(&camera, &camInfo, &embeddedInfo)){
        printf("Failed to initialize PointGrey camera\n");
        return -1;
    }

	// set camera to be software trigger mode (software trigger latency: 50 us)
	if (!pointgrey_set_triggermode(&camera, &triggerMode)){
		printf("Failed to set camera to be software trigger\n");
		return -1;
	}
	else
		printf("Camera is now in software trigger mode ... \n");
		
	// configurate camera
	double cam_shutter_time; // [ms]
	std::ifstream lastfile("../temp/shutter_time.txt");
  	if (lastfile.is_open()){
		printf("Use previous shutter time in 'shutter_time.txt'\n");
		lastfile >> cam_shutter_time;
    	lastfile.close();
    } else {
		printf("Unable to open file '../temp/shutter_time.txt'; using default 1 ms\n");
		cam_shutter_time = 1.0;
	}
	
    // define capture image pointer
	int SENSOR_WIDTH, SENSOR_HEIGHT;
    pointgrey_get_sensor_size(&camInfo, SENSOR_WIDTH, SENSOR_HEIGHT);
    cv::Mat image_cap(cv::Size(SENSOR_WIDTH, SENSOR_HEIGHT), CV_8UC1, cv::Scalar(0));
    
    // set capture image size
	if (!pointgrey_set_capture_size(&camera, SENSOR_WIDTH, SENSOR_HEIGHT))
	{
	    printf("Failed to set PointGrey camera ROI area.\n");
		return -1;
	}
    // =======================================================
    
    // start capture
	if(!pointgrey_start_capture(&camera))
	{
	    printf("Failed to start PointGrey camera capture.\n");
        return -1;
	}
    
#ifdef SOFTWARE_TRIGGER_CAMERA
    if (!CheckSoftwareTriggerPresence(&camera))
    {
        std::cout << "SOFT_ASYNC_TRIGGER not implemented on this camera! Stopping "
                "application" << std::endl;
        return -1;
    }
#else
    std::cout << "Trigger the camera by sending a trigger pulse to GPIO"
    		  << triggerMode.source << std::endl;
#endif

    float cam_frame_rate = 30.0f; // [Hz]
    pointgrey_set_property(&camera, (float)cam_shutter_time, cam_frame_rate);
    
    // =============   Camera Settings (END)   =============

    // initialize CUDA OpenGL interop; register the resource
    if (cuda_gl_interop_setup_texture(texture)){
        printf("Cannot setup CUDA OpenGL interop; program failed.\n");
        return -1;
    }
    
    // temporary cuda Array
    cudaArray *cuArray;

    // initialize the black screen
    cuda2gl(d_SLM_img, cuArray, SLM_WIDTH, SLM_HEIGHT);

    // set V-sync for the SLM
    glfwSwapInterval(swap_interval);
    
    static int swap_tear = (glfwExtensionSupported("WGL_EXT_swap_control_tear") ||
                            glfwExtensionSupported("GLX_EXT_swap_control_tear"));

    glfwSetFramebufferSizeCallback(windows[0], framebuffer_size_callback);

    // show black image
    draw_quad(texture);
    glfwSwapBuffers(windows[0]);

#ifdef SOFTWARE_TRIGGER_CAMERA
        // Check that the trigger is ready
        PollForTriggerReady(&camera);

        // Fire software trigger
        if (!FireSoftwareTrigger(&camera))
        {
            printf("\nError firing software trigger\n");
            return -1;
        }
#endif

    // capture loop
    int num_frame = 1000;
    int cntr = 0;
    for (int i = 0; i < num_frame; i++)
    {
        // Get the image
        error = camera.RetrieveBuffer( &rawImage );
        if ( error != PGRERROR_OK )
        {
            error.PrintErrorTrace();
            continue;
        }
    	printf("Capturing %d/%d\n", i+1, num_frame);

#ifdef SOFTWARE_TRIGGER_CAMERA
        // Check that the trigger is ready
        PollForTriggerReady(&camera);

        // Fire software trigger
        if (!FireSoftwareTrigger(&camera))
        {
            printf("\nError firing software trigger\n");
            return -1;
        }
#endif

        // set image title name; read the calibration image
        if (i < num_frame/2 - 1){
            sprintf(image_name, "../projects/AO_xy_alignment/data/pla_%d.png", ++cntr);
        }
        else if (i == num_frame/2 - 1){
            sprintf(image_name, "../projects/AO_xy_alignment/data/pla_%d.png", ++cntr);
            
            // load square image to host
            h_SLM_img = cv::imread("../projects/AO_xy_alignment/cali_microlens_img.png", CV_LOAD_IMAGE_GRAYSCALE);
            
            // check if the image has been loaded
    		if ( h_SLM_img.empty() )
    		{
        		printf("Could not open the microlens array image; Please run gen_microlensarray.m first!\n");
        		return -1;
    		}

            // copy from host to device
            checkCudaErrors(cudaMemcpy(d_SLM_img, h_SLM_img.ptr(), dataSize_int, cudaMemcpyHostToDevice));
            
            // write to OpenGL texture
            cuda2gl(d_SLM_img, cuArray, SLM_WIDTH, SLM_HEIGHT);

        	// show the images using OpenGL
        	if (!glfwWindowShouldClose(windows[0]))
        	{
            	draw_quad(texture);
            	glfwSwapBuffers(windows[0]);
        	}
    		printf("Swap now ----------------------------------\n");
		}
		else
			sprintf(image_name, "../projects/AO_xy_alignment/data/square_%d.png", ++cntr);

        // convert to OpenCV Mat
        image_cap.data = rawImage.GetData();

        // save captured image
        cv::imwrite(image_name, image_cap);
    }
    
    // cleanup
    checkCudaErrors(cudaFree(d_SLM_img));
    
    cudaDeviceReset();

    // stop camera capture
    if (!pointgrey_stop_capture(&camera)){
        printf("Stop PointGrey capture failed.\n");
        return -1;
    }

    glfwTerminate();
    exit(EXIT_SUCCESS);
}
