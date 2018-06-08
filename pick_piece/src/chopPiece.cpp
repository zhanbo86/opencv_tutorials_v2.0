#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <time.h>

using namespace cv;
using namespace std;
#include <stdio.h>

int num_p = 0;
int x_int = 0;
int y_int = 0;
std::vector<int> x_temp(6);
std::vector<int> y_temp(6);
bool capture = false;
void my_mouse_callback(int event,int x,int y,int flags,void* param)
{
  switch (event) {
  case CV_EVENT_LBUTTONDOWN:{
    capture = true;
    x_int = x;
    y_int = y;
//    std::cout<<"x_int = "<<x_int<<"\t"<<"y_int = "<<y_int<<std::endl;
//    std::cout<<"capture_ = "<<capture_<<std::endl;
  }
  break;
  case CV_EVENT_LBUTTONUP:{
    if(capture)
    {
      capture = false;
//      std::cout<<"x_int = "<<x_int<<"\t"<<"y_int = "<<y_int<<std::endl;
      x_temp.at(num_p) = x_int;
      y_temp.at(num_p) = y_int;
      num_p++;
//      std::cout<<"num_p="<<num_p<<"\t"<<"capture_ = "<<capture_<<std::endl;
//      std::cout<<"x_[num_p] = "<<x_temp.at(num_p-1)<<"\t"<<"y_[num_p] = "<<y_temp.at(num_p-1)<<std::endl;
    }
  }
  break;
  }
}


int main()
{
    //load raw img data.
    std::cout<<"load img from raw_img file."<<std::endl;
    Mat img = imread("/home/zb/BoZhan/picking_test/scene5/10.png",
                     CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
    Mat mask = Mat::zeros(img.size(),CV_8UC1);
    cout<<"img.depth = "<<img.depth()<<"\t"<<"img.channels = "<<img.channels()<<endl;
    cout<<"img.size = "<<img.cols<<" * "<<img.rows<<endl;


    ////pick ocr piece
    imshow( "img", img );
    cvSetMouseCallback("img",my_mouse_callback,NULL);
    cv::Mat chop_piece;
    while(num_p!=6)
    {
      imshow( "img", img );
      cvWaitKey(10);
    }
    if(num_p==6)
    {
//      std::vector<Point2f> obj_corners(6);
//      obj_corners[0] = Point( x_temp[0], y_temp[0] );
//      obj_corners[1] = Point( x_temp[1], y_temp[1] );
//      obj_corners[2] = Point( x_temp[2], y_temp[2] );
//      obj_corners[3] = Point( x_temp[3], y_temp[3] );
//      obj_corners[4] = Point( x_temp[4], y_temp[4] );
//      obj_corners[5] = Point( x_temp[5], y_temp[5] );
//      Rect roi_rect = Rect(obj_corners[0].x,obj_corners[0].y,
//                           obj_corners[1].x-obj_corners[0].x,
//                           obj_corners[3].y-obj_corners[0].y);
//      img(roi_rect).copyTo(chop_piece);
//      line( img , obj_corners[0], obj_corners[1], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
//      line( img , obj_corners[1], obj_corners[2], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
//      line( img , obj_corners[2], obj_corners[3], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
//      line( img , obj_corners[3], obj_corners[0], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );

      Point root_points[1][6];
      root_points[0][0] = Point( x_temp[0], y_temp[0] );
      root_points[0][1] = Point( x_temp[1], y_temp[1] );
      root_points[0][2] = Point( x_temp[2], y_temp[2] );
      root_points[0][3] = Point( x_temp[3], y_temp[3] );
      root_points[0][4] = Point( x_temp[4], y_temp[4] );
      root_points[0][5] = Point( x_temp[5], y_temp[5] );
      const Point* ppt[1] = {root_points[0]};
      int npt[] = {6};
      polylines(img, ppt, npt, 1, 1, Scalar(255),1,8,0);
      polylines(mask, ppt, npt, 1, 1, Scalar(255),1,8,0);
      imshow("img", img);
      waitKey();
      fillPoly(mask, ppt, npt, 1, Scalar(255));
      imshow("Test", mask);
      waitKey();
    }



   //// identifing single characters
    const char* chop_piece_folder = "/home/zb/BoZhan/picking_test/scene5/";
    std::stringstream ss(std::stringstream::in | std::stringstream::out);
    ss << chop_piece_folder << "/mask" << 10<<".jpg";
    imwrite(ss.str(),mask);
    return 0;
}


