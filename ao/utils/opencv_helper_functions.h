#ifndef OPENCV_HELPER_FUNCTIONS_H
#define OPENCV_HELPER_FUNCTIONS_H

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

#define M_microlens_width  10 
#define M_microlens_height 9

// function to get perspective matrix from file
cv::Mat cal_perspective_matrix(int x_crop, int y_crop)
{
	std::ifstream SLMfile;
	float temp[2*M_microlens_width*M_microlens_height];
	
	// load in the SLM data
	cv::Point2f SLM_points[M_microlens_width*M_microlens_height];
	SLMfile.open("../temp/cali_SLM_centers.txt");
  	if (SLMfile.is_open()){
		for (int i = 0; i < 2*M_microlens_width*M_microlens_height; i++)
			SLMfile >> temp[i];
    	SLMfile.close();
    } else
		printf("Unable to open file '../temp/cali_SLM_centers.txt'\n");
	for (int i = 0; i < M_microlens_width*M_microlens_height; i++)
		SLM_points[i] = cv::Point2f(temp[i], temp[i+M_microlens_width*M_microlens_height]);

	// get the sensor point grid (with cropping compensation)
	cv::Point2f sensor_points[M_microlens_width*M_microlens_height];
	SLMfile.open("../temp/cali_sensor_centers.txt");
  	if (SLMfile.is_open()){
		for (int i = 0; i < 2*M_microlens_width*M_microlens_height; i++)
			SLMfile >> temp[i];
    	SLMfile.close();
    } else
		printf("Unable to open file '../temp/cali_sensor_centers_quick.txt'\n");
	for (int i = 0; i < M_microlens_width*M_microlens_height; i++)
		sensor_points[i] = cv::Point2f(temp[i] - x_crop, temp[i+M_microlens_width*M_microlens_height] - y_crop);

	// fit the perspective matrix G0
	cv::Mat G0 = cv::getPerspectiveTransform(sensor_points, SLM_points);

	// print the calibration matrix
	std::cout << "Perspective matrix G0 = " << std::endl << G0 << std::endl;
	
	return G0;
}
#endif

