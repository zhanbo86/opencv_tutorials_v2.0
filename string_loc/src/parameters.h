#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;


double cameraMatrix_00 = 3.5503121613717367e+03;
double cameraMatrix_01 = 0.0;
double cameraMatrix_02 = 1.2732676523578607e+03;
double cameraMatrix_10 = 0.0;
double cameraMatrix_11 = 3.5470311346721546e+03;
double cameraMatrix_12 = 1.0562383749330231e+03;
double cameraMatrix_20 = 0.0;
double cameraMatrix_21 = 0.0;
double cameraMatrix_22 = 1.0;

double distCoeff_00 = -1.5060857208196030e-01;
double distCoeff_01 = -6.0931821333313424e-03;
double distCoeff_02 = -1.2465297521024776e-03;
double distCoeff_03 = -1.0107849173074991e-03;
double distCoeff_04 = 0.0;


float cam_field_width = 141;
float cam_field_height = 127;
float cam_pixel_width = 2448;
float cam_pixel_height = 2048;
int string_pos_dx = 5;
int string_pos_dy = 5;

////plate one
//float plate_width = 94.02;
//float plate_height = 71.70;
//float string_x = 16.09;
//float string_y = 64.8;
//float string_width = 4.14;
//float string_height = 1.48;


//plate two
float plate_width = 86.05;
float plate_height = 72.14;
float string_x = 73.83;
float string_y = 50.28;
float string_width = 2.06;
float string_height = 3.76;


int plate_color = 1;
/*****
  1:black
  2:blue
  3:white
  4:red
*****/

#endif // PARAMETERS_H
