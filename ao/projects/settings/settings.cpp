#if defined(LINUX32) || defined(LINUX64)
#define LINUX
#endif

// for GUI
#define CVUI_DISABLE_COMPILATION_NOTICES
#define CVUI_IMPLEMENTATION // start from cvui 2.5.0-BETA
#include "cvui.h"

// standard libraries
#include <stdlib.h>

// flycapture SDK
#include <FlyCapture2.h>

// OpenCV
#include <opencv2/opencv.hpp>

// input/output std library
#include <iostream>
#include <fstream>

// for software trigger
#ifdef LINUX
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#endif

// glfw
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// use software trigger for the camera sync
// (uncomment to enable hardware trigger)
#define SOFTWARE_TRIGGER_CAMERA

// our project utilities
#include "flycapture2_helper_functions.h"
#include "glfw_helper_functions.h"

using namespace FlyCapture2;

#define CALIBRATION_WINDOW_NAME		"Calibration"

/////////////////////////////////////////////////////////////////////////////////
///// application entry point
/////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    // ===============   Define Camera   ===============
    Error error;
    Camera camera;                      // camera
    CameraInfo camInfo;                 // camera information
    EmbeddedImageInfo embeddedInfo;     // embedded settings
    TriggerMode triggerMode;			// camera trigger mode
    
    // initialize the camera
    if (!pointgrey_initialize(&camera, &camInfo, &embeddedInfo)){
        printf("Failed to initialize PointGrey camera\n");
        return -1;
    }

	// set camera trigger mode (unknown software trigger latency)
	if (!pointgrey_set_triggermode(&camera, &triggerMode))
	{
		printf("Failed to set camera to be software trigger\n");
		return -1;
	}
	else
		printf("Camera is now in software trigger mode ... \n");
		
    // get sensor size
    int SENSOR_WIDTH, SENSOR_HEIGHT;
    pointgrey_get_sensor_size(&camInfo, SENSOR_WIDTH, SENSOR_HEIGHT);

	// set frame rate & trigger delay
    float cam_frame_rate = 30.0f; // Hz
    float trigger_delay  = 0.0f; // s
    
    // image dimensions
    int M_width  = SENSOR_WIDTH;
    int M_height = SENSOR_HEIGHT;
    
	// set capture image size
	if (!pointgrey_set_capture_size(&camera, M_width, M_height))
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

    // =================   GLFW Settings   =================
    
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
    
    // write down the SLM size
    std::ofstream tempfile;
  	tempfile.open("../temp/SLM_sizes.txt");
  	tempfile << SLM_WIDTH  << "\n";
  	tempfile << SLM_HEIGHT << "\n";
  	tempfile.close();
    // ===========   GLFW Settings (END)  ===========
    
    // define variables for the camera
    Image rawImage;
    cv::Mat image_ref(cv::Size(M_width, M_height), CV_8UC1, cv::Scalar(0));
    
	// define shutter time and read from last time
	double shutter_time;
	std::ifstream lastfile;
    lastfile.open("../temp/shutter_time.txt");
  	if (lastfile.is_open()){
		lastfile >> shutter_time;
    	lastfile.close();
    } else {
		printf("Unable to open file in '../temp/shutter_time.txt'; using default 1 ms\n");
		shutter_time = 1.0;
	}

	// define shutter time and the background GUI
	cv::Mat frame = cv::Mat(cv::Size(300, 150), CV_8UC3);

	// create OpenCV windows
	cv::namedWindow(CALIBRATION_WINDOW_NAME);
	cv::moveWindow(CALIBRATION_WINDOW_NAME, 0, 0);
	cv::namedWindow("Calibration Image", 0);
	cv::moveWindow("Calibration Image", 0, 200);
    
	cvui::init(CALIBRATION_WINDOW_NAME);

    // the calibration loop
	printf("\nUser: Adjust the shutter time to do the calibration ... \n\n");
	while (true)
	{
		// draw the frame
		frame = cv::Scalar(49, 52, 49);
		cvui::text(frame, 30, 30, "Sensor shutter time [ms]");
		cvui::trackbar(frame, 30, 50, 200, &shutter_time, 0.1, 50.0);					
		cvui::printf(frame, frame.cols - 160, frame.rows - 30, 0.4, 0xCECECE, "VCC Imaging @ KAUST");
		if (cvui::button(frame, 30, 110, "Done")) {
			break;
		}

		// round shutter time to 0.1 precision
		shutter_time = (shutter_time*10.0f) / 10.0f;

    	// set sensor shutter time
        pointgrey_set_property(&camera, (float)shutter_time, cam_frame_rate, trigger_delay);

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

		// capture camera image
        error = camera.RetrieveBuffer( &rawImage );
        if ( error != PGRERROR_OK )
        {
            printf("Capture error\n");
            continue;
        }

        // convert to OpenCV Mat
        image_ref.data = rawImage.GetData();
		
		// show captured image
		cv::imshow("Calibration Image", image_ref);

		cvui::update();
		cv::imshow(CALIBRATION_WINDOW_NAME, frame);
		cv::waitKey(1);
	}

	// write to .txt and save the shutter time for next use
  	tempfile.open ("../temp/shutter_time.txt");
  	tempfile << std::fixed << std::setprecision(1) << shutter_time;
  	tempfile.close();

	// erase the image window
	cv::destroyWindow("Calibration Image");
	cv::destroyWindow(CALIBRATION_WINDOW_NAME);

    // stop camera capture
    if (!pointgrey_stop_capture(&camera, &triggerMode)){
        printf("Stop PointGrey capture failed.\n");
        return -1;
    }
	printf("Done.\n");

    glfwTerminate();
    
    exit(EXIT_SUCCESS);
}
