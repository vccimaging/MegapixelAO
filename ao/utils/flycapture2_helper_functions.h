#ifndef FLYCAPTURE2_HELPER_FUNCTIONS_H
#define FLYCAPTURE2_HELPER_FUNCTIONS_H

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif 

#include <stdlib.h>
#include <stdio.h>
#include <sstream>

#include <FlyCapture2.h>

#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"


void PrintBuildInfo()
{
    FlyCapture2::FC2Version fc2Version;
    FlyCapture2::Utilities::GetLibraryVersion(&fc2Version);

    std::ostringstream version;
    version << "FlyCapture2 library version: " << fc2Version.major << "."
            << fc2Version.minor << "." << fc2Version.type << "."
            << fc2Version.build;
    std::cout << version.str() << std::endl;

    std::ostringstream timeStamp;
    timeStamp << "Application build date: " << __DATE__ << " " << __TIME__;
    std::cout << timeStamp.str() << std::endl << std::endl;
}


void PrintCameraInfo(FlyCapture2::CameraInfo *pCamInfo)
{
    std::cout << std::endl;
    std::cout << "*** CAMERA INFORMATION ***" << std::endl;
    std::cout << "Serial number - " << pCamInfo->serialNumber << std::endl;
    std::cout << "Camera model - " << pCamInfo->modelName << std::endl;
    std::cout << "Camera vendor - " << pCamInfo->vendorName << std::endl;
    std::cout << "Sensor - " << pCamInfo->sensorInfo << std::endl;
    std::cout << "Resolution - " << pCamInfo->sensorResolution << std::endl;
    std::cout << "Firmware version - " << pCamInfo->firmwareVersion << std::endl;
    std::cout << "Firmware build time - " << pCamInfo->firmwareBuildTime << std::endl
         << std::endl;
}


static bool pointgrey_initialize(FlyCapture2::Camera *camera, FlyCapture2::CameraInfo *camInfo, 
                                 FlyCapture2::EmbeddedImageInfo *embeddedInfo)
{
    FlyCapture2::Error error;
    
    // Connect the camera
    error = camera->Connect( 0 );
    if ( error != FlyCapture2::PGRERROR_OK )
    {
        printf("Failed to connect to camera\n");
        return false;
    }

    // Power on the camera
	const unsigned int k_cameraPower = 0x610;
	const unsigned int k_powerVal = 0x80000000;
	error  = camera->WriteRegister( k_cameraPower, k_powerVal );
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

    const unsigned int millisecondsToSleep = 100;
	unsigned int regVal = 0;
	unsigned int retries = 10;

    // Wait for camera to complete power-up
    do
    {
#if defined(_WIN32) || defined(_WIN64)
        Sleep(millisecondsToSleep);
#elif defined(LINUX)
        struct timespec nsDelay;
        nsDelay.tv_sec = 0;
        nsDelay.tv_nsec = (long)millisecondsToSleep * 1000000L;
        nanosleep(&nsDelay, NULL);
#endif
        error = camera->ReadRegister(k_cameraPower, &regVal);
        if (error == FlyCapture2::PGRERROR_TIMEOUT)
        {
            // ignore timeout errors, camera may not be responding to
            // register reads during power-up
        }
        else if (error != FlyCapture2::PGRERROR_OK)
        {
			error.PrintErrorTrace();
            return false;
        }

        retries--;
    } while ((regVal & k_powerVal) == 0 && retries > 0);

	// Check for timeout errors after retrying
	if (error == FlyCapture2::PGRERROR_TIMEOUT)
	{
		error.PrintErrorTrace();
		return false;
	}

    // Get the camera info and print it out
    error = camera->GetCameraInfo( camInfo );
    if ( error != FlyCapture2::PGRERROR_OK )
    {
        error.PrintErrorTrace();
        return false;
    }
	PrintCameraInfo(camInfo);
    return true;
}


// software trigger functions
static bool CheckSoftwareTriggerPresence( FlyCapture2::Camera* pCam )
{
	const unsigned int k_triggerInq = 0x530;

	FlyCapture2::Error error;
	unsigned int regVal = 0;

	error = pCam->ReadRegister( k_triggerInq, &regVal );

	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}

	if( ( regVal & 0x10000 ) != 0x10000 )
		return false;

	return true;
}


static bool PollForTriggerReady( FlyCapture2::Camera* pCam )
{
	const unsigned int k_softwareTrigger = 0x62C;
	FlyCapture2::Error error;
	unsigned int regVal = 0;

	do{
		error = pCam->ReadRegister( k_softwareTrigger, &regVal );
		if (error != FlyCapture2::PGRERROR_OK)
		{
			error.PrintErrorTrace();
			return false;
		}

	} while ( (regVal >> 31) != 0 );
	
	return true;
}


static bool FireSoftwareTrigger( FlyCapture2::Camera* pCam )
{
	const unsigned int k_softwareTrigger = 0x62C;
	const unsigned int k_fireVal = 0x80000000;
	FlyCapture2::Error error;

	error = pCam->WriteRegister( k_softwareTrigger, k_fireVal );
	if (error != FlyCapture2::PGRERROR_OK)
	{
		error.PrintErrorTrace();
		return false;
	}
	
	return true;
}


static bool pointgrey_start_capture(FlyCapture2::Camera *camera, FlyCapture2::TriggerMode* triggerMode = NULL)
{
    FlyCapture2::Error error = camera->StartCapture();
    if (error == FlyCapture2::PGRERROR_ISOCH_BANDWIDTH_EXCEEDED)
    {
        error.PrintErrorTrace();
        return false;
    }
    else if (error != FlyCapture2::PGRERROR_OK)
    {
        error.PrintErrorTrace();
        return false;
    }

    if (triggerMode != NULL){
#ifdef SOFTWARE_TRIGGER_CAMERA
	if (!CheckSoftwareTriggerPresence(camera))
	{
		std::cout << "SOFT_ASYNC_TRIGGER not implemented on this camera! Stopping application" << std::endl;
		return false;
	}
#else
	std::cout << "Trigger the camera by sending a trigger pulse to GPIO" << triggerMode->source << std::endl;
#endif
    }
    
	return true;
}


static bool pointgrey_stop_capture(FlyCapture2::Camera *camera, FlyCapture2::TriggerMode* triggerMode = NULL)
{
    FlyCapture2::Error error;
    
	// Turn trigger mode off
	if (triggerMode != NULL)
	{
	    triggerMode->onOff = false;
	    error = camera->SetTriggerMode( triggerMode );
	    if (error != FlyCapture2::PGRERROR_OK)
	    {
            error.PrintErrorTrace();
		    return false;
	    }
    }

    // Stop capturing images
    error = camera->StopCapture();
    if ( error != FlyCapture2::PGRERROR_OK )
    {
//         This may fail when the camera was removed, so don't show 
//         an error message
    }

    // Disconnect the camera
	error = camera->Disconnect();
	if (error != FlyCapture2::PGRERROR_OK)
	{
        error.PrintErrorTrace();
        return false;
	}
	
	return true;
}


static bool pointgrey_set_triggermode(FlyCapture2::Camera* camera, FlyCapture2::TriggerMode* triggerMode)
{
    FlyCapture2::Error error;
    
#ifndef SOFTWARE_TRIGGER_CAMERA
	// Check for external trigger support
	FlyCapture2::TriggerModeInfo triggerModeInfo;
	error = camera->GetTriggerModeInfo( &triggerModeInfo );
	if (error != FlyCapture2::PGRERROR_OK){
        error.PrintErrorTrace();
		return false;
	}
	if ( triggerModeInfo.present != true ){
		std::cout << "Camera does not support external trigger! Exiting..." << std::endl;
		return false;
	}
#endif
	
	// Get current trigger settings
	error = camera->GetTriggerMode( triggerMode );
	if (error != FlyCapture2::PGRERROR_OK){
		printf("Failed to get current trigger settings\n");
		return false;
	}

	// Set camera to trigger mode 14
	triggerMode->onOff = true;
	triggerMode->mode = 14; // for mode 15, software trigger is disabled.
	triggerMode->parameter = 0;

#ifdef SOFTWARE_TRIGGER_CAMERA
	// A source of 7 means software trigger
	triggerMode->source = 7;
#else
	// Triggering the camera externally using source 3 (GPIO3).
	triggerMode->source = 3;
#endif

	error = camera->SetTriggerMode( triggerMode );
	if (error != FlyCapture2::PGRERROR_OK)
	{
        error.PrintErrorTrace();
		return false;
	}

	// Get trigger delay info
	FlyCapture2::TriggerDelayInfo triggerDelayInfo;
	error = camera->GetTriggerDelayInfo( &triggerDelayInfo );
	if (error != FlyCapture2::PGRERROR_OK){
        error.PrintErrorTrace();
		return false;
	}
	if ( triggerDelayInfo.present != true ){
		std::cout << "Read trigger delay info failed! Exiting..." << std::endl;
		return false;
	}

    // Poll to ensure camera is ready
	bool retVal = PollForTriggerReady( camera );
	if( !retVal )
	{
		std::cout << std::endl;
		std::cout << "Error polling for trigger ready!" << std::endl;
		return false;
	}

    return true;
}


static void pointgrey_set_property(FlyCapture2::Camera* camera, float shutter_time, float frame_rate, float trigger_delay = 0.0f)
{
	FlyCapture2::FC2Config BufferFrame;
	camera->GetConfiguration(&BufferFrame);
	BufferFrame.numBuffers = 1;
//	BufferFrame.grabMode = FlyCapture2::BUFFER_FRAMES; // use frame buffer
	BufferFrame.grabMode = FlyCapture2::DROP_FRAMES;   // stream mode: get only the newest image
	BufferFrame.highPerformanceRetrieveBuffer = true;
	camera->SetConfiguration(&BufferFrame);
	
    // ===============   Camera Setups   ===============
	// Configurate camera settings
    FlyCapture2::Property property;

    // (i) Set Gamma curve to be 1
    property.type = FlyCapture2::GAMMA;
    camera->GetProperty(&property);
    property.onOff = true;
    property.absValue = 1.0f;
    camera->SetProperty(&property);

    // (ii) Set Shutter to be [shutter_time] ms
    property.type = FlyCapture2::SHUTTER;
    camera->GetProperty(&property);
    property.autoManualMode = false;
    property.absValue = shutter_time;
    camera->SetProperty(&property);

    // (iii) Set Gain to be 0 dB
    property.type = FlyCapture2::GAIN;
    camera->GetProperty(&property);
    property.autoManualMode = false;
    property.absValue = 0.0f;
    camera->SetProperty(&property);
    
    // (iv) Set frame rate to be [frame_rate] fps
    property.type = FlyCapture2::FRAME_RATE;
    camera->GetProperty(&property);
    property.onOff = true;
    property.autoManualMode = false;
    property.absValue = frame_rate;
    camera->SetProperty(&property);
    
    // (v) Set trigger delay to be [trigger_delay] s
    property.type = FlyCapture2::TRIGGER_DELAY;
    camera->GetProperty(&property);
    property.onOff = true;
    property.autoManualMode = false;
    property.absValue = trigger_delay;
    camera->SetProperty(&property);
}


static void pointgrey_get_sensor_size(FlyCapture2::CameraInfo *camInfo, int& width_sensor, int& height_sensor)
{
    // phrase the sensor info to constants
	std::string sensorResolution;
	sensorResolution = camInfo->sensorResolution;
	width_sensor = std::stoi(sensorResolution);
	sensorResolution.erase(0,5);
 	height_sensor = std::stoi(sensorResolution);
}


static void PrintFormat7Capabilities(FlyCapture2::Format7Info* fmt7Info)
{
    std::cout << "Max image pixels: (" << fmt7Info->maxWidth << ", "
         << fmt7Info->maxHeight << ")" << std::endl;
    std::cout << "Image Unit size: (" << fmt7Info->imageHStepSize << ", "
         << fmt7Info->imageVStepSize << ")" << std::endl;
    std::cout << "Offset Unit size: (" << fmt7Info->offsetHStepSize << ", "
         << fmt7Info->offsetVStepSize << ")" << std::endl;
    std::cout << "Pixel format bitfield: 0x" << fmt7Info->pixelFormatBitField << std::endl;
}


static bool pointgrey_set_capture_size(FlyCapture2::Camera* camera, int& width_image, int& height_image)
{
    FlyCapture2::Error error;
    
	const FlyCapture2::Mode k_fmt7Mode = FlyCapture2::MODE_0;
    const FlyCapture2::PixelFormat k_fmt7PixFmt = FlyCapture2::PIXEL_FORMAT_MONO8;
	
	// Query for available Format 7 modes
    FlyCapture2::Format7Info fmt7Info;
    bool supported;
    fmt7Info.mode = k_fmt7Mode;
    error = camera->GetFormat7Info(&fmt7Info, &supported);
    if (error != FlyCapture2::PGRERROR_OK)
    {
		error.PrintErrorTrace();
        return false;
    }

    PrintFormat7Capabilities(&fmt7Info);

    if ((k_fmt7PixFmt & fmt7Info.pixelFormatBitField) == 0)
    {
        // Pixel format not supported!
        printf("Pixel format is not supported\n");
        return false;
    }

    FlyCapture2::Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = k_fmt7Mode;
    fmt7ImageSettings.width  = width_image;
    fmt7ImageSettings.height = height_image;
    fmt7ImageSettings.offsetX = (fmt7Info.maxWidth- width_image) / 2;
    fmt7ImageSettings.offsetY = (fmt7Info.maxHeight - height_image) / 2;
    fmt7ImageSettings.pixelFormat = k_fmt7PixFmt;

    bool valid;
    FlyCapture2::Format7PacketInfo fmt7PacketInfo;

    // Validate the settings to make sure that they are valid
    error = camera->ValidateFormat7Settings(
        &fmt7ImageSettings, &valid, &fmt7PacketInfo);
    if (error != FlyCapture2::PGRERROR_OK)
    {
		error.PrintErrorTrace();
        return false;
    }

    if (!valid)
    {
        // Settings are not valid
        printf("Format7 settings are not valid\n");
        return false;
    }

    // Set the settings to the camera
    error = camera->SetFormat7Configuration(
        &fmt7ImageSettings, fmt7PacketInfo.recommendedBytesPerPacket);
    if (error != FlyCapture2::PGRERROR_OK)
    {
		error.PrintErrorTrace();
        return false;
    }
    
    return true;
}

#endif

