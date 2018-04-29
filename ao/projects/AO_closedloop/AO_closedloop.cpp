#if defined(LINUX32) || defined(LINUX64)
#define LINUX
#endif

// for GUI
#define CVUI_DISABLE_COMPILATION_NOTICES
#define CVUI_IMPLEMENTATION // start from cvui 2.5.0-BETA
#include "cvui.h"

// Please keep this include order to make sure a successful compilation!
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <cuda_profiler_api.h> // for profiling
#include <helper_cuda.h> 

// flycapture SDK
#include <FlyCapture2.h>

// OpenCV
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

// OpenGL+GLFW
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// put CUDA-OpenGL interop head after Glad and GLFW
#include <cuda_gl_interop.h>

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#include <cufft.h>

typedef float real;
typedef cufftComplex complex;

// use software trigger for the camera sync
// (uncomment to enable hardware trigger)
// #define SOFTWARE_TRIGGER_CAMERA

#include "common.h"
#include "gen_SLM_images.h"
#include "flowCUDA.h"
#include "prepare_cufft_warmup.h"
#include "prepare_precomputations.h"

// our project utilities
#include "cuda_gl_interop_helper_functions.h"
#include "glfw_helper_functions.h"
#include "flycapture2_helper_functions.h"
#include "IO_helper_functions.h"

using namespace FlyCapture2;

#define CALIBRATION_WINDOW_NAME		"Calibration"

// flag for visualization
#define FLAG_VISUAL 1

// boundary condition
static int BOUND_COND = 1;
// 0: ZERO
// 1: REPLICATE

// GLFW variables and CUDA+OpenGL interop variable
static GLFWwindow* windows[1];

// swap buffer parameters
static int swap_interval = 1;

// define resize factor for imshow (for speed)
static int show_factor = 2;

// apply ROF denoising or not
static bool isrof = false;

/////////////////////////////////////////////////////////////////////////////////
///// application entry point
/////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // ===============   Define Camera   ===============
    Error error;
    Camera camera;                      // camera
    CameraInfo camInfo;                 // camera information
    EmbeddedImageInfo embeddedInfo;     // embedded settings
    TriggerMode triggerMode;			// camera trigger mode
    
    // initialize the camera
    if (!pointgrey_initialize(&camera, &camInfo, &embeddedInfo))
    {
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

    // define telescope
    float telescope = 200.0f / 150.0f; // SLM to wavefront sensor
    
    // compute scaling factor
    float scale_factor = 0.05f * telescope; // our estimation
    
	// set frame rate & trigger delay
    float cam_frame_rate = 30.0f; // Hz
    float trigger_delay  = 0/120.0f; // s
    
    // image dimensions
    int M_width  = 992; // has to be multiple of 32: to suit texture memory
    int M_height = 992;
    int M_stride = iAlignUp(M_width);
    
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
    std::cout << "Trigger the camera by sending a trigger pulse to GPI3"
    		  << triggerMode.source << std::endl;
#endif

    // ===============   CPU & GPU Setups   ===============
    // welcome message
    printf("%s Starting...\n\n", "cuda_1storderDCT pyramid");

    // pick GPU
    findCudaDevice(argc, (const char **)argv);

    // remove CUDA timing latency
	cudaFree(0);
    cudaSetDevice(0);  // use GeForce GTX TITAN

    // define unknown size
    int N_width  = 1024;
    int N_height = 1024;
    
    // get extra dimensions
    const static int L_width  = (N_width  - M_width) /2;
    const static int L_height = (N_height - M_height)/2;
    
    // define cropping coordinate
    const static int x_crop = (SENSOR_WIDTH  - M_width) /2;
    const static int y_crop = (SENSOR_HEIGHT - M_height)/2;
    const static int x_crop_phase = (SENSOR_WIDTH  - N_width) /2;
    const static int y_crop_phase = (SENSOR_HEIGHT - N_height)/2;

    // print the cropping area origin
    printf("x: %d, y: %d \n", x_crop, y_crop);

    // host point for reference image and measurement image, and the result
    float *h_phi = new float [N_width * N_height];

    // =================   OpenGL Setups   =================
    
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
    windows[0] = open_window("SLM phase (wrapped)", NULL, monitors[SLM_no], 0, 0);
    if (!windows[0])
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(windows[0]);
    glViewport(0, 0, SLM_WIDTH, SLM_HEIGHT);
    
    // set V-sync for the SLM
    glfwSwapInterval(swap_interval);
    static int swap_tear = (glfwExtensionSupported("WGL_EXT_swap_control_tear") ||
                            glfwExtensionSupported("GLX_EXT_swap_control_tear"));

    // load shaders
	GLuint program = LoadShader("../src/background_SLM.vert", 
		    					"../src/background_SLM.frag");

	// get matrix G0
	cv::Mat G0(3, 3, CV_32F);
	std::ifstream file("../temp/perspective_T.txt");
  	if (file.is_open()){
		for (int i = 0; i < 9; i++)
			file >> G0.at<float>(i);
    	file.close();
	}
	std::cout << "Perspective matrix G0 = " << std::endl << G0 << std::endl;
	
    // prepare VAO
    GLuint VBO, VAO, EBO;

	// flip
    prepare_VBO_VAO_EBO(VBO, VAO, EBO, 1.0f, 1.0f, 0.0f, 0.0f,
                                       0.0f, 1.0f, 1.0f, 0.0f);

	// create texture on OpenGL
	GLuint tex_rec = create_texture(SLM_WIDTH, SLM_HEIGHT, NULL, BOUND_COND);        // our reconstruction
    GLuint tex_ref = create_texture(SLM_WIDTH, SLM_HEIGHT, NULL, BOUND_COND);    // the ground truth (if any)
	
	// prepare the texture positions (please be sure to follow the *.frag) for the fragment shader
    prepare_texture_shader(program);
    
    // define SLM image host pointer (in cv::Mat)
    cv::Mat SLM_img(cv::Size(N_width, N_height), CV_8UC1, cv::Scalar(0));
    cv::Mat SLM_img_whole(cv::Size(SLM_WIDTH, SLM_HEIGHT), CV_8UC1, cv::Scalar(0));

	// device pointer (in GpuMat) for holding the SLM image
    cv::cuda::GpuMat temp_gpumat(SLM_img_whole.size(), CV_32F);
    cv::cuda::GpuMat SLM_gpumat (SLM_img_whole.size(), CV_32F);
    temp_gpumat.setTo(cv::Scalar::all(0));
    SLM_gpumat.setTo(cv::Scalar::all(0));
    
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
    
    // initialize tex_rec with zeros; for the calibration
    cuda2gl(d_SLM_img, cuArray, SLM_WIDTH, SLM_HEIGHT);
    
	// initialize CUDA OpenGL interop; register the resource tex_ref
    if (cuda_gl_interop_setup_texture(tex_ref))
    {
        printf("Cannot setup CUDA OpenGL interop; program failed.\n");
        return -1;
    }

    // initialize tex_ref with zeros; for the calibration
    cuda2gl(d_SLM_img, cuArray, SLM_WIDTH, SLM_HEIGHT);
    
    // show a black image on the SLM
    draw_quad_shader(program, VAO, tex_rec, tex_ref);
    glfwSwapBuffers(windows[0]);

    // for frame counts
    unsigned long frame_count = 0;

    // ===========   GLFW Settings (END)  ===========
    
    // ============   Wavefront Sensor Calibration   ============

    // define variables for the camera
    Image rawImage;
    cv::Mat image_ref(cv::Size(M_width, M_height), CV_8UC1, cv::Scalar(0));
    cv::Mat image_cap(cv::Size(M_width, M_height), CV_8UC1, cv::Scalar(0));
    cv::Mat image_tmp(cv::Size(M_width, M_height), CV_32F,  cv::Scalar(0));
    cv::Mat phase_crop, phase_crop_color, phase_show, phase_show_color;
    phase_crop = cv::Mat(N_height, N_width, CV_32F, cv::Scalar(0));
    phase_show = cv::Mat(N_height/show_factor, N_width/show_factor, CV_32F, cv::Scalar(0));
    phase_show_color = cv::Mat(N_height/show_factor, N_width/show_factor, CV_8UC3, cv::Scalar(0));
    
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
	std::ofstream tempfile;
  	tempfile.open ("../temp/shutter_time.txt");
  	tempfile << std::fixed << std::setprecision(1) << shutter_time;
  	tempfile.close();

	// erase the image window
	cv::destroyWindow("Calibration Image");
	cv::destroyWindow(CALIBRATION_WINDOW_NAME);

    // crop and copy the reference image
    if (image_ref.empty())
	{
		printf("Reference Image Failed!\n");
		return false;
	}

    // set the reference image
	image_ref.convertTo(image_ref, CV_32F); // Have to do convertTo after cv::Rect!

	printf("Calibration done!\n\n");

	// =============   Load SLM Ground Truth   =============
	printf("Loading SLM ground truth data ...  ");
	
	// load SLM ground truth to tex_ref
    // (you can set a different initial phase, rather than zero phase here)
    SLM_img_whole = cv::imread("../projects/AO_closedloop/SLM_gt_pla_0.png", CV_LOAD_IMAGE_GRAYSCALE);
//    SLM_img_whole = cv::imread("../projects/AO_closedloop/SLM_gt_spherical_4e-05.png", CV_LOAD_IMAGE_GRAYSCALE);

	// check if the image has been loaded
    if (SLM_img_whole.empty())
    {
    	printf("Could not load the reference image; Please check your directory!\n");
        return -1;
    }
    checkCudaErrors(cudaMemcpy(d_SLM_img, SLM_img_whole.ptr(0), 
                               SLM_WIDTH*SLM_HEIGHT*sizeof(unsigned char), cudaMemcpyHostToDevice));
    cuda2gl(d_SLM_img, cuArray, SLM_WIDTH, SLM_HEIGHT);
    
    // reset d_SLM_img to 127 (this is the mean when we are using in gen_SLM_images.cu function)
    checkCudaErrors(cudaMemset(d_SLM_img, 0x7F, SLM_WIDTH*SLM_HEIGHT*sizeof(unsigned char)));

    // show the SLM ground truth to the screen
    glfwMakeContextCurrent(windows[0]);
    draw_quad_shader(program, VAO, tex_rec, tex_ref);
    glfwSwapBuffers(windows[0]);
 
	printf("Done.\n");

	// ===========   Load SLM Ground Truth (Done)   ===========

    // register the resource to tex_rec
    if (cuda_gl_interop_setup_texture(tex_rec))
    {
    	printf("Cannot setup CUDA OpenGL interop; program failed.\n");
        return -1;
    }
    
    // ===============   Algorithm Setups   ===============

    // set bilateral filter parameters
	int kernel_size = 15;
	int kernel_size_u = 15;
	float sigma_pixel = 150.0f;
	float sigma_space = 150.0f;
	
	// SLM phase wrap
	float SLM_wrap = 1.0f;
	
    // AO gain
    float AO_gain = 1.0f / telescope / SLM_wrap; // 6 pi SLM phase wrap; but in reality it is ~ 3 pi; so stable when < 2/3

    // smoothness
    const float alpha = 3.0f;

    // number of pyramid levels
    const int nLevels = 2;
    int endLevel = 0; // set the pyramid end level
    
    // number of warping iterations (from coarse to fine)
    int *nWarpIters = new int [nLevels];
    nWarpIters[0] = 2;
    nWarpIters[1] = 1;
    nWarpIters[2] = 2;
    nWarpIters[3] = 2;

    // proximal parameter (changable if using half-quadratic-splitting)
    float *mu = new float [nLevels];
    mu[0] = 100.0f;
    mu[1] = 20.0f;
    mu[2] = 20.0f;
    mu[3] = 20.0f;

    // number of proximal algorithm iterations
    int nAlgoIters = 10;

    // define variables for proximal algorithm
    float **pI0 = new float *[nLevels];
    float **pI1 = new float *[nLevels]; // store the image pyramid
    const float **mat_x_hat = new const float *[nLevels];
    float **d_I0_coeff = new float *[nLevels]; // store the cubic coefficients image pyramid
    
    int *pW_N = new int [nLevels];
    int *pH_N = new int [nLevels];
    int *pS_N = new int [nLevels];
    int *pW_M = new int [nLevels];
    int *pH_M = new int [nLevels];
    int *pS_M = new int [nLevels];
    int *pW_L = new int [nLevels];
    int *pH_L = new int [nLevels];
    
    float **phi = new float *[nLevels];
    float **phi_delta = new float *[nLevels];
    float **w_x = new float *[nLevels];
    float **w_y = new float *[nLevels];
    float **zeta_x = new float *[nLevels];
    float **zeta_y = new float *[nLevels];

    float **w11 = new float *[nLevels];
    float **w12_or_w22 = new float *[nLevels];
    float **w13 = new float *[nLevels];
    float **w21 = new float *[nLevels];
    float **w23 = new float *[nLevels];
    
    const complex **ww_1 = new const complex *[nLevels];
    const complex **ww_2 = new const complex *[nLevels];
    
    float **temp_x = new float *[nLevels];
    float **temp_y = new float *[nLevels];
    complex **temp_dct = new complex *[nLevels];
    
    int dataSize_M = M_width*M_height*sizeof(float);
    
    // prepare pyramid
    checkCudaErrors(cudaMalloc(pI0 + nLevels-1, dataSize_M));
    checkCudaErrors(cudaMalloc(pI1 + nLevels-1, dataSize_M));
    checkCudaErrors(cudaMalloc(d_I0_coeff + nLevels-1, dataSize_M));

    // pass calibration data in the device poiners
    checkCudaErrors(cudaMemcpy((void *)pI0[nLevels-1], image_ref.ptr<float>(0), dataSize_M, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *)d_I0_coeff[nLevels-1], image_ref.ptr<float>(0), dataSize_M, cudaMemcpyHostToDevice));

	// ============ do ROF denoising on the images ============
	// allocate device pointers
    float *temp_IM, *px, *py, *temp_px, *temp_py, *temp;
	checkCudaErrors(cudaMalloc(&temp_IM, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&px, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&py, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp_px, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp_py, M_width*M_height*sizeof(float)));
	checkCudaErrors(cudaMalloc(&temp, M_width*M_height*sizeof(float)));
	
	float *phi_center;
	checkCudaErrors(cudaMalloc(&phi_center, M_width*M_height*sizeof(float)));
	
	// set ROF denoising parameters
	float theta = 0.125f;
	int iter = 20;
	float alp = 0.95f;

	// run FISTA ROF on reference image
	if (isrof){
	fista_rof(pI0[nLevels-1], temp_IM, px, py, temp_px, temp_py, temp, M_width, M_height, theta, iter, alp);
	fista_rof(d_I0_coeff[nLevels-1], temp_IM, px, py, temp_px, temp_py, temp, M_width, M_height, theta, iter, alp);
	}
	
	// =========================================================
	
    // determine L value for best performance (we use mod 2 here)
    for (int i = nLevels-1; i >= 0; i--){
        pW_L[i] = L_width  >> (nLevels-1 - i);
        pH_L[i] = L_height >> (nLevels-1 - i);
        if (pW_L[i] < 2)    pW_L[i] = 2;
        if (pH_L[i] < 2)    pH_L[i] = 2;
    }
    
    pW_N[nLevels-1] = M_width  + 2*pW_L[nLevels-1];
    pH_N[nLevels-1] = M_height + 2*pH_L[nLevels-1];
    pS_N[nLevels-1] = iAlignUp(pW_N[nLevels-1]);
    pW_M[nLevels-1] = M_width;
    pH_M[nLevels-1] = M_height;
    pS_M[nLevels-1] = M_stride;

    printf("initial sizes: W_N = %d, H_N = %d, S_N = %d\n", N_width, N_height, iAlignUp(pW_N[nLevels-1]));
    printf("initial sizes: W_M = %d, H_M = %d, S_M = %d\n", M_width, M_height, M_stride);
    
	// pre-computations
    prepare_precomputations(N_width, N_height, nLevels-1, 
                            pW_N, pH_N, pS_N, 
                            pW_M, pH_M, pS_M, 
                            pW_L, pH_L, pI0, pI1, d_I0_coeff, alpha, mu, 
                            mat_x_hat, ww_1, ww_2);
                            
    // allocate memory for the variables
    printf("\nAllocating device variables ...  ");
    int dataSize_N;
    for (int i = nLevels-1; i >= 0; i--){
    
        dataSize_N = pW_N[i] * pH_N[i] * sizeof(float);
        dataSize_M = pW_M[i] * pH_M[i] * sizeof(float);
            
        checkCudaErrors(cudaMalloc(phi + i, dataSize_N));
        
        checkCudaErrors(cudaMalloc(phi_delta + i, dataSize_N));
        checkCudaErrors(cudaMalloc(w_x + i, dataSize_N));
        checkCudaErrors(cudaMalloc(w_y + i, dataSize_N));
        checkCudaErrors(cudaMalloc(zeta_x + i, dataSize_N));   
        checkCudaErrors(cudaMalloc(zeta_y + i, dataSize_N));

        checkCudaErrors(cudaMalloc(w11 + i, dataSize_M));
        checkCudaErrors(cudaMalloc(w13 + i, dataSize_M));
        checkCudaErrors(cudaMalloc(w12_or_w22 + i, dataSize_M));
        checkCudaErrors(cudaMalloc(w21 + i, dataSize_M));
        checkCudaErrors(cudaMalloc(w23 + i, dataSize_M));
                        
        checkCudaErrors(cudaMalloc(temp_x + i, dataSize_N));
        checkCudaErrors(cudaMalloc(temp_y + i, dataSize_N));
        checkCudaErrors(cudaMalloc(temp_dct + i, 
                        pW_N[i]*pH_N[i]*sizeof(complex)));
    }
    printf("Done.\n");
    
    // create OpenCV GpuMat to hold the reconstructed phase
    cv::cuda::GpuMat phi_gpumat;
    phi_gpumat.upload(phase_crop);
    
    // prepare cufft plans & warmup
    printf("Preparing CuFFT plans and warmups ...  ");
	cufftHandle plan_dct_1, plan_dct_2,
	            plan_dct_3, plan_dct_4,
	            plan_dct_5, plan_dct_6,
	            plan_dct_7, plan_dct_8;
	int Length1[1], Length2[1];
	Length1[0] = pH_N[0]; // for each FFT, the Length1 is N_height
	Length2[0] = pW_N[0];  // for each FFT, the Length2 is N_width
	cufftPlanMany(&plan_dct_1, 1, Length1, 
					  Length1, pW_N[0], 1, 
					  Length1, pW_N[0], 1, 
					CUFFT_C2C, pW_N[0]);
	cufftPlanMany(&plan_dct_2, 1, Length2, 
					  Length2, pH_N[0], 1, 
					  Length2, pH_N[0], 1, 
					CUFFT_C2C, pH_N[0]);
	Length1[0] = pH_N[1]; // for each FFT, the Length1 is N_height
	Length2[0] = pW_N[1];  // for each FFT, the Length2 is N_width
	cufftPlanMany(&plan_dct_3, 1, Length1, 
					  Length1, pW_N[1], 1, 
					  Length1, pW_N[1], 1, 
					CUFFT_C2C, pW_N[1]);
	cufftPlanMany(&plan_dct_4, 1, Length2,
					  Length2, pH_N[1], 1, 
					  Length2, pH_N[1], 1, 
					CUFFT_C2C, pH_N[1]);
	Length1[0] = pH_N[2]; // for each FFT, the Length1 is N_height
	Length2[0] = pW_N[2];  // for each FFT, the Length2 is N_width				
	cufftPlanMany(&plan_dct_5, 1, Length1, 
					  Length1, pW_N[2], 1, 
					  Length1, pW_N[2], 1, 
					CUFFT_C2C, pW_N[2]);
	cufftPlanMany(&plan_dct_6, 1, Length2, 
					  Length2, pH_N[2], 1, 
					  Length2, pH_N[2], 1, 
					CUFFT_C2C, pH_N[2]);
	Length1[0] = pH_N[3]; // for each FFT, the Length1 is N_height
	Length2[0] = pW_N[3];  // for each FFT, the Length2 is N_width			
	cufftPlanMany(&plan_dct_7, 1, Length2, 
					  Length1, pW_N[3], 1, 
					  Length1, pW_N[3], 1, 
					CUFFT_C2C, pW_N[3]);
	cufftPlanMany(&plan_dct_8, 1, Length2, 
					  Length2, pH_N[3], 1, 
					  Length2, pH_N[3], 1, 
					CUFFT_C2C, pH_N[3]);
					
	// cufft warmup
	complex *h_warmup_in  = new complex [N_width * N_height];
	complex *h_warmup_out = new complex [N_width * N_height];
	cufft_warper(h_warmup_in, N_width, N_height, plan_dct_1, h_warmup_out);
	cufft_warper(h_warmup_in, N_width, N_height, plan_dct_2, h_warmup_out);
	delete [] h_warmup_in;
	delete [] h_warmup_out;
    printf("Done.\n");

	// ==========================  Test variables ==========================
	// test variable
    int test_width  = N_width;
    int test_height = N_height;
    float *h_test = new float [test_width * test_height];
	
	float *phase_acc;
	checkCudaErrors(cudaMalloc(&phase_acc, test_width*test_height*sizeof(float)));
    checkCudaErrors(cudaMemset(phase_acc, 0, test_width*test_height*sizeof(float)));
	// =======================  Test variables (END) =======================

    // define cuda profiling interface
//	cudaProfilerStart();
	GpuTimer timer;
	
    // for image/data savings
    int cntr = 0;

	// define default visualization option:
	// = 1:interference rings
	// = 2: height map
	// = 3: height map (wrapped) 
	int visual_opt = 1;
	float height_map_max = 0.0f;
    cv::Mat lut = cv::imread("../projects/AO_closedloop/coolwarm.png", CV_LOAD_IMAGE_COLOR);

	// initialize frame
	frame = cv::Mat(190, 530, CV_8UC3);
	bool checked1 = true;
	bool checked2 = false;
	bool checked3 = false;
	bool checked4 = true;
	bool checked5 = false;
	bool checked6 = false;
	double min_phase = -1.0f;
	double max_phase =  1.0f;

	// initial state of checked boxes
	bool checked1_last = checked1;
	bool checked2_last = checked2;
	bool checked3_last = checked3;
	bool checked4_last = checked4;
	bool checked5_last = checked5;
	bool checked6_last = checked6;

	bool checked_AO = false;

	if (FLAG_VISUAL)
	{
        // define windows
        cv::namedWindow("Reconstructed Wavefront", CV_WINDOW_AUTOSIZE);
        cv::moveWindow("Reconstructed Wavefront", 0, 0);
        cv::namedWindow("Captured Image", CV_WINDOW_AUTOSIZE);
        cv::moveWindow("Captured Image", 0, 0);
    }
    {
	    cv::namedWindow("Options");
	    cvui::init("Options");
    }
	
    // ===================   Main Loop   ===================
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
    float RMS = 100.0f;
    float *RMS_SLM = new float [100000];
    float *RMS_wavefront = new float [100000];
    int counter_RMS = 0;
    float acc_elapsed_time = 0.0f;
    char string_show[80];
    float fps = 0.0f;
    while( true )
    {
    	// timing started
		timer.Start();
		
	    // get the image
        error = camera.RetrieveBuffer( &rawImage );
        if ( error != PGRERROR_OK )
        {
            printf("Capture error\n");
            continue;
        }

        // convert to OpenCV Mat
        image_cap.data = rawImage.GetData();
        
        // crop the received image
        image_cap.convertTo(image_tmp, CV_32F);
        
    	// copy data from host to device and prepare the pyramid
    	checkCudaErrors(cudaMemcpy((void *)pI1[nLevels-1], image_tmp.ptr<float>(0), M_width*M_height*sizeof(float), 		cudaMemcpyHostToDevice));

		// run FISTA ROF on captured image
		if (isrof)
			fista_rof(pI1[nLevels-1], temp_IM, px, py, temp_px, temp_py, temp, M_width, M_height, theta, iter, alp);

	    // ============== To be optimized ==============
        // first-order ADMM+DCT algorithm
        ComputeFlowCUDA(pI0, pI1, N_width, N_height, 
                        M_width, M_height,
                        pW_N, pH_N, pS_N,
                        pW_M, pH_M, pS_M,
                        alpha, mu,
                        nWarpIters, nAlgoIters, nLevels, endLevel,
                        plan_dct_1, plan_dct_2,
                        plan_dct_3, plan_dct_4,
                        plan_dct_5, plan_dct_6,
                        plan_dct_7, plan_dct_8,
                        d_I0_coeff,
                        phi, phi_delta, w_x, w_y,
                        zeta_x, zeta_y, mat_x_hat,
                        w11, w13, w12_or_w22, 
                        w21, w23, ww_1, ww_2,
                        temp_x, temp_y, temp_dct, RMS, N_width*N_height,
                        phi_center,
                        h_test,
                        scale_factor);

		// ================= Applying bilateral filtering =================
		// copy device pointer to GpuMat (can be further optimized into flowCUDA function)
        checkCudaErrors(cudaMemcpy(phi_gpumat.ptr(), phi[nLevels-1], N_width*N_height*sizeof(float), cudaMemcpyDeviceToDevice));

		// bilateral filtering on the phase; reduce noise and for better wrapping results
		switch (BOUND_COND){
			case 0:{
				cv::cuda::bilateralFilter(phi_gpumat, phi_gpumat, kernel_size, sigma_pixel, sigma_space, cv::BORDER_CONSTANT);
				break;
			}
			case 1:{
				cv::cuda::bilateralFilter(phi_gpumat, phi_gpumat, kernel_size, sigma_pixel, sigma_space, cv::BORDER_REPLICATE);
				break;
			}
		}
		// ============== Applying bilateral filtering (END) ==============

		cntr++;
    	frame = cv::Scalar(49, 52, 49);
    	
        // ============================ SLM ============================
        if (cvui::checkbox(frame, 130, 155, "AO On", &checked_AO))
        if (!(cntr % 3)) 
        // every 3 frames update once; it has to be tuned so as to suit 
        // your SLM refresh speed
        {
            // update SLM
            gen_SLM_img(d_SLM_img, phi_gpumat, temp_gpumat, SLM_gpumat,
            			N_width, N_height, SLM_WIDTH, SLM_HEIGHT, scale_factor, AO_gain, kernel_size_u, G0, RMS_SLM[counter_RMS], BOUND_COND);
        
            // copy to OpenGL texture and show on the SLM
            cuda2gl(d_SLM_img, cuArray, SLM_WIDTH, SLM_HEIGHT);

            // show SLM
            glfwMakeContextCurrent(windows[0]);
            if (!glfwWindowShouldClose(windows[0]))
            {
                draw_quad_shader(program, VAO, tex_rec, tex_ref);
                glfwSwapBuffers(windows[0]);
            }
		
			// record wavefront RMS
            RMS_wavefront[counter_RMS] = RMS;
            counter_RMS++;
        }
        
        if (cvui::button(frame, 250, 150, "Reset SLM"))
        {
            // reset SLM to zero for reference
            checkCudaErrors(cudaMemset(d_SLM_img, 0, SLM_WIDTH*SLM_HEIGHT*sizeof(unsigned char)));
    		SLM_gpumat.setTo(cv::Scalar::all(0));
            
            // copy to OpenGL texture and show on the SLM
            cuda2gl(d_SLM_img, cuArray, SLM_WIDTH, SLM_HEIGHT);

            // reset SLM to mean 127 for our solver
            checkCudaErrors(cudaMemset(d_SLM_img, 0x7F, SLM_WIDTH*SLM_HEIGHT*sizeof(unsigned char)));
            
            // show SLM
            glfwMakeContextCurrent(windows[0]);
            draw_quad_shader(program, VAO, tex_rec, tex_ref);
            glfwSwapBuffers(windows[0]);
        }
        // ============================ SLM ============================

        // ============================ GUI ============================
        // our marker
        cvui::printf(frame, frame.cols - 160, frame.rows - 30, 0.4, 0xCECECE, "VCC Imaging @ KAUST");
	    	
        // set GUI
		if (FLAG_VISUAL)
		{
		    cvui::text(frame, 30, 30, "Colormap");
		    cvui::checkbox(frame, 30, 60, "Grayscale", &checked4);
		    cvui::checkbox(frame, 30, 90, "Jet", &checked5);
		    cvui::checkbox(frame, 30, 120, "Coolwarm", &checked6);

		    cvui::text(frame, 130, 30, "Visualization Type");
		    cvui::checkbox(frame, 130, 60, "Interference Rings", &checked1);
		    cvui::checkbox(frame, 130, 90, "Height Map", &checked2);
		    cvui::checkbox(frame, 130, 120, "Height Map (wrapped)", &checked3);

		    if (checked2){
			    cvui::printf(frame, 300, 70, 0.4, 0xCECECE, "Min [lambda]");
			    cvui::printf(frame, 420, 70, 0.4, 0xCECECE, "Max [lambda]");
			    cvui::counter(frame, 300, 90, &min_phase, 0.5, "%.1f");
			    cvui::counter(frame, 420, 90, &max_phase, 0.5, "%.1f");
		    }

		    // check status
		    if (checked1 > checked1_last){
			    checked2 = false;
			    checked3 = false;
		    } else if (checked1 < checked1_last){
			    checked1 = true;
		    } else{
			    if (checked2 > checked2_last){
			    	checked1 = false;
			    	checked3 = false;
			    } else if (checked2 < checked2_last){
			    	checked2 = true;
			    } else{
			    	if (checked3 > checked3_last){
			    		checked1 = false;
			    		checked2 = false;
			    	} else if (checked3 < checked3_last){
			    		checked3 = true;
			    		}
		 	    }
		    }

		    if (checked4 > checked4_last){
			    checked5 = false;
			    checked6 = false;
		    } else if (checked4 < checked4_last){
			    checked4 = true;
		    } else{
			    if (checked5 > checked5_last){
			    	checked4 = false;
			    	checked6 = false;
			    } else if (checked5 < checked5_last){
			    	checked5 = true;
			    } else{
			    	if (checked6 > checked6_last){
			    		checked4 = false;
			    		checked5 = false;
			    	} else if (checked6 < checked6_last){
			    		checked6 = true;
			    	}
		 	    }
		    }

    		// update status
	    	checked1_last = checked1;
	    	checked2_last = checked2;
	    	checked3_last = checked3;
	    	checked4_last = checked4;
	    	checked5_last = checked5;
	    	checked6_last = checked6;

	    	// prevent min > max
	    	if (min_phase > max_phase)
	    		min_phase = max_phase;

	    	// set visualization option
	    	if (checked1_last == true)	visual_opt = 1;
	    	if (checked2_last == true)	visual_opt = 2;
	    	if (checked3_last == true)	visual_opt = 3;
        }

		if (cvui::button(frame, 30, 150, "Quit"))
			break;
		
        // update the GUI
		cvui::update();
        // ============================ GUI ============================
		
		// ========================== Visualization ==========================
        // show captured result
		if (FLAG_VISUAL && fps != 0.0f)
		{
        	// convert data for visualization
        	visualize_phase(phi_gpumat.ptr<float>(), phi_gpumat.ptr<float>(), pW_N[nLevels-1], pH_N[nLevels-1], 
            	            visual_opt, scale_factor, static_cast<int>(min_phase), static_cast<int>(max_phase));

	        // copy data from device to host
		    phi_gpumat.download(phase_crop);

			// prepare string for show					
			sprintf(string_show, "FPS %.2f  RMS %.2f lambda", fps, RMS);

			// apply colormap and show phase image
			if (checked4_last)
			{
		    	cv::resize(phase_crop(cv::Rect(L_width, L_height, M_width, M_height)), phase_show, phase_show.size(), 0, 0, cv::INTER_LINEAR);
				cv::putText(phase_show,string_show,
							cv::Point(phase_show.cols-250,phase_show.rows-10), // Coordinates
            				cv::FONT_HERSHEY_PLAIN, // Font
            				1, // Scale. 2.0 = 2x bigger
            				cv::Scalar(0,0,0), // Color
            				1); // Anti-alias
		    	cv::imshow("Reconstructed Wavefront", phase_show);
			}

			if (checked5_last)
			{
            	phase_crop.convertTo(phase_crop, CV_8UC1, 255.0);
		    	cv::applyColorMap(phase_crop, phase_crop_color, cv::COLORMAP_JET);
		    	cv::resize(phase_crop_color(cv::Rect(L_width, L_height, M_width, M_height)), phase_show_color, phase_show_color.size(), 0, 0, cv::INTER_LINEAR);
				cv::putText(phase_show_color,string_show,
							cv::Point(phase_show_color.cols-250,phase_show_color.rows-10), // Coordinates
            				cv::FONT_HERSHEY_PLAIN, // Font
            				1, // Scale. 2.0 = 2x bigger
            				cv::Scalar(0,0,0), // Color
            				1); // Anti-alias
		    	cv::imshow("Reconstructed Wavefront", phase_show_color);
        	}
        	
			if (checked6_last)
			{
				phase_crop.convertTo(phase_crop, CV_8UC1, 255.0);
				cvtColor(phase_crop.clone(), phase_crop, cv::COLOR_GRAY2BGR);
				cv::LUT(phase_crop, lut, phase_crop_color);
		    	cv::resize(phase_crop_color(cv::Rect(L_width, L_height, M_width, M_height)), phase_show_color, phase_show_color.size(), 0, 0, cv::INTER_LINEAR);
				cv::putText(phase_show_color,string_show,
							cv::Point(phase_show_color.cols-250,phase_show_color.rows-10), // Coordinates
            				cv::FONT_HERSHEY_PLAIN, // Font
            				1, // Scale. 2.0 = 2x bigger
            				cv::Scalar(0,0,0), // Color
            				1); // Anti-alias
		    	cv::imshow("Reconstructed Wavefront", phase_show_color);
			}

			// show raw data
			cv::resize(image_cap, phase_show, phase_show.size(), 0, 0, cv::INTER_LINEAR);
			cv::imshow("Captured Image", phase_show);
        }
		// ========================== Visualization ==========================

		// show everything on the control (primary) screen
		cv::imshow("Options", frame);
		cv::waitKey(1); // wait for SLM response time and visualization

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

		// timing ended
		timer.Stop();
		fps = 1000/timer.Elapsed();
		
		// print info
		if (cntr > 1)
			acc_elapsed_time += timer.Elapsed();
		printf("Phase RMS: %.4f lambda. Elapsed time: %.4f ms. Frame rate: %.4f fps. \n", 
		    									RMS, timer.Elapsed(), fps);
    }

	// =============================== Report ===============================
	// print RMS record
	float wavefront_variance = 0.0f;
	printf("Wavefront RMS:\n");
	for (int i = 0; i < counter_RMS; i++)
	{
		printf("%f\n", RMS_wavefront[i]);
		wavefront_variance += RMS_wavefront[i];
	}
	float wavefront_mean = wavefront_variance / (float)(counter_RMS);
	wavefront_variance = 0.0f;
	for (int i = 0; i < counter_RMS; i++)
		wavefront_variance += (RMS_wavefront[i] - wavefront_mean) * (RMS_wavefront[i] - wavefront_mean);
	
	// print wavefront RMS variance
	wavefront_variance = sqrtf( wavefront_variance / (float)(counter_RMS) );
	printf("Wavefront variance: %f lambda\n", wavefront_variance);
	
    acc_elapsed_time /= cntr;
	printf("Average elapsed time: %f ms; Average frame rate: %f fps\n", acc_elapsed_time, 1000/acc_elapsed_time);
    
    // save accumulated phase
    checkCudaErrors(cudaMemcpy(h_phi, phase_acc, N_width*N_height*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Total frames: (cntr) = %d\n", cntr);
    
    // write results
	checkCudaErrors(cudaMemcpy(h_test, phi_gpumat.ptr<float>(), test_width*test_height*sizeof(float), cudaMemcpyDeviceToHost));
    WriteFloFile("../projects/AO_closedloop/FlowGPU.flo", test_width, test_height, h_test, h_test);

    // cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteTextures(1, &tex_rec);
    glDeleteTextures(1, &tex_ref);
    
	checkCudaErrors(cudaFree(phi_center));
    
    // free resources
    delete[] h_phi;
    delete[] h_test;
	checkCudaErrors(cudaFree(phase_acc));

	delete[] RMS_SLM;
	delete[] RMS_wavefront;

    // cleanup
    for (int i = 0; i < nLevels; i++)
    {
        checkCudaErrors(cudaFree((void *)pI0[i]));
        checkCudaErrors(cudaFree((void *)pI1[i]));
        checkCudaErrors(cudaFree((void *)d_I0_coeff[i]));
        
        checkCudaErrors(cudaFree((void *)mat_x_hat[i]));
        checkCudaErrors(cudaFree((void *)ww_1[i]));
        checkCudaErrors(cudaFree((void *)ww_2[i]));
        
        checkCudaErrors(cudaFree((void *)phi[i]));
        checkCudaErrors(cudaFree((void *)phi_delta[i]));
        checkCudaErrors(cudaFree((void *)w_x[i]));
        checkCudaErrors(cudaFree((void *)w_y[i]));
        checkCudaErrors(cudaFree((void *)zeta_x[i]));
        checkCudaErrors(cudaFree((void *)zeta_y[i]));
        
        checkCudaErrors(cudaFree((void *)w11[i]));
        checkCudaErrors(cudaFree((void *)w13[i]));
        checkCudaErrors(cudaFree((void *)w12_or_w22[i]));
        checkCudaErrors(cudaFree((void *)w21[i]));
        checkCudaErrors(cudaFree((void *)w23[i]));
        
        checkCudaErrors(cudaFree((void *)temp_x[i]));
        checkCudaErrors(cudaFree((void *)temp_y[i]));
        checkCudaErrors(cudaFree((void *)temp_dct[i]));
    }

    delete[] pI0;
    delete[] pI1;
    delete[] mat_x_hat;
    delete[] d_I0_coeff;
    
    delete[] ww_1;
    delete[] ww_2;
    delete[] pW_N;
    delete[] pH_N;
    delete[] pS_N;
    delete[] pW_M;
    delete[] pH_M;
    delete[] pS_M;
    delete[] pW_L;
    delete[] pH_L;
    
    delete[] nWarpIters;
    delete[] mu;
    
    delete[] phi;
    delete[] phi_delta;
    delete[] w_x;
    delete[] w_y;
    delete[] zeta_x;
    delete[] zeta_y;
    
    delete[] w11;
    delete[] w13;
    delete[] w12_or_w22;
    delete[] w21;
    delete[] w23;
    
    delete[] temp_x;
    delete[] temp_y;
    delete[] temp_dct;
    
    checkCudaErrors(cudaFree(d_SLM_img));
    
    // destroy
	timer.Free();
	cufftDestroy(plan_dct_1);
	cufftDestroy(plan_dct_2);
	cufftDestroy(plan_dct_3);
	cufftDestroy(plan_dct_4);
	cufftDestroy(plan_dct_5);
	cufftDestroy(plan_dct_6);
	cufftDestroy(plan_dct_7);
	cufftDestroy(plan_dct_8);
	
	// clean ROF denoising pointers
	checkCudaErrors(cudaFree(temp_IM));
	checkCudaErrors(cudaFree(px));
	checkCudaErrors(cudaFree(py));
	checkCudaErrors(cudaFree(temp_px));
	checkCudaErrors(cudaFree(temp_py));
	checkCudaErrors(cudaFree(temp));
	
    // stop camera capture
    if (!pointgrey_stop_capture(&camera)){
        printf("Stop PointGrey capture failed.\n");
        return -1;
    }
    
    cudaDeviceReset();
    
    glfwDestroyWindow(windows[0]);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
