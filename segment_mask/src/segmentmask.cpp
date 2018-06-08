#include "segmentmask.h"
#include <cv_bridge/cv_bridge.h>
//#include <sensor_msgs/image_encodings.h>
//#include <iostream>
//#include <sstream>
//#include <string>
//#include <ctime>
//#include <cstdio>
//#include <time.h>
//#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <ros/ros.h>


#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <time.h>
#include <stdio.h>

using namespace cv;
using namespace std;



bool segment_img (segment_mask::recognition::Request &req, segment_mask::recognition::Response &res)
{
  std::cout<<"I have received a image!!!"<<std::endl;
  Mat mask = imread("/home/zb/BoZhan/OpenCV/Opencv_tutorials/pictures/piece0.jpg",
                   CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);

  cout<<"mask.depth = "<<mask.depth()<<"\t"<<"mask.channels = "<<mask.channels()<<endl;
  cout<<"mask.size = "<<mask.cols<<" * "<<mask.rows<<endl;

  Mat dst;
  threshold(mask, dst, 0, 1, THRESH_BINARY);
//  imshow("mask",dst);
//  waitKey(6000);

  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", dst).toImageMsg();


////  cv_bridge::CvImage img_bridge;
////  sensor_msgs::Image img_msg;
////  std_msgs::Header header; // empty header
//////  header.seq = counter; // user defined counter
////  header.stamp = ros::Time::now(); // time
////  img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, mask);
////  img_bridge.toImageMsg(img_msg); // from cv_bridge to sensor_msgs::Image

  std::vector<std_msgs::Int32> obj_num_shoot(1);
  obj_num_shoot.at(0).data = 3;
  res.im_out = *img_msg;
  res.sort = obj_num_shoot;
  std::cout<<"return mask!!!"<<std::endl;
  return true;
}



int main (int argc, char** argv)
{
  ros::init (argc, argv, "segment");
  ros::NodeHandle nh;
  ros::Rate loop_rate(200);

  //server, communication between shooting and recognition
  ros::ServiceServer segment_mask = nh.advertiseService("recognition",segment_img);

  while(ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return (0);
}


