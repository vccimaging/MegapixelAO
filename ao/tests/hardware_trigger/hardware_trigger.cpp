#if defined(LINUX32) || defined(LINUX64)
#define LINUX
#endif

#ifdef LINUX
#include <time.h>
#include <unistd.h>
#endif

// std library
#include <iostream>

// use software trigger for the camera sync
// (uncomment to enable hardware trigger)
// #define SOFTWARE_TRIGGER_CAMERA

// our functions
#include "flycapture2_helper_functions.h"

using namespace FlyCapture2;
using namespace std;

int main(int argc, char **argv)
{
    PrintBuildInfo();

    const int k_numImages = 10;

    Error error;

    BusManager busMgr;
    unsigned int numCameras;
    error = busMgr.GetNumOfCameras(&numCameras);
    if (error != PGRERROR_OK)
    {
		error.PrintErrorTrace();
        return -1;
    }

    cout << "Number of cameras detected: " << numCameras << endl;

    if (numCameras < 1)
    {
        cout << "Insufficient number of cameras... exiting" << endl;
        return -1;
    }

    PGRGuid guid;
    error = busMgr.GetCameraFromIndex(0, &guid);
    if (error != PGRERROR_OK)
    {
		error.PrintErrorTrace();
        return -1;
    }

	// define camera
    Camera cam;                       // camera
    CameraInfo camInfo;               // camera information
    EmbeddedImageInfo embeddedInfo;   // embedded settings
    TriggerMode triggerMode;		  // camera trigger mode
    
    // initialize the camera
    if (!pointgrey_initialize(&cam, &camInfo, &embeddedInfo)){
        printf("Failed to initialize PointGrey camera\n");
        return -1;
    }

#ifdef SOFTWARE_TRIGGER_CAMERA
	// set camera to be software trigger mode
	if (!pointgrey_set_triggermode(&cam, &triggerMode)){
		printf("Failed to set camera to be software trigger\n");
		return -1;
	}
	else
		printf("Camera is now in software trigger mode ... \n");
#endif

    // Get the camera configuration
    FC2Config config;
    error = cam.GetConfiguration(&config);
    if (error != PGRERROR_OK)
    {
		error.PrintErrorTrace();
        return -1;
    }

    // Set the grab timeout to 5 seconds
    config.grabTimeout = 5000;

    // Set the camera configuration
    error = cam.SetConfiguration(&config);
    if (error != PGRERROR_OK)
    {
		error.PrintErrorTrace();
        return -1;
    }

    // Camera is ready, start capturing images
    error = cam.StartCapture();
    if (error != PGRERROR_OK)
    {
		error.PrintErrorTrace();
        return -1;
    }

#ifdef SOFTWARE_TRIGGER_CAMERA
    if (!CheckSoftwareTriggerPresence(&cam))
    {
        cout << "SOFT_ASYNC_TRIGGER not implemented on this camera! Stopping "
                "application"
             << endl;
        return -1;
    }
#else
    cout << "Trigger the camera by sending a trigger pulse to GPIO3, mode:"
         << triggerMode.source << endl;
#endif

    Image image;
    for (int imageCount = 0; imageCount < k_numImages; imageCount++)
    {
#ifdef SOFTWARE_TRIGGER_CAMERA
        // Check that the trigger is ready
        PollForTriggerReady(&cam);

        cout << "Press the Enter key to initiate a software trigger" << endl;
        cin.ignore();

        // Fire software trigger
        bool retVal = FireSoftwareTrigger(&cam);
        if (!retVal)
        {
            cout << endl;
            cout << "Error firing software trigger" << endl;
            return -1;
        }
#endif

        // Grab image
        error = cam.RetrieveBuffer(&image);
        if (error != PGRERROR_OK)
        {
			error.PrintErrorTrace();
            break;
        }
        cout << "." << endl;
    }

    // stop camera capture
    if (!pointgrey_stop_capture(&cam)){
        printf("Stop PointGrey capture failed.\n");
        return -1;
    }

    cout << "Done! Press Enter to exit..." << endl;
    cin.ignore();

    return 0;
}

