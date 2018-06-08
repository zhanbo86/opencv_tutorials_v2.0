//#include<opencv2/highgui.hpp>
//#include<opencv2/core.hpp>
//#include<opencv2/opencv.hpp>

//#include <opencv2/core/utility.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/videoio.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>



#include "codebook.h"

const double PI=3.1415926;


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void MaximumChannel(IplImage* img)
{
  for(int y=0;y<img->height;y++)
  {
    uchar* ptr = (uchar*)(img->imageData + y*img->widthStep);
    for(int x=0;x<img->width;x++)
    {
      *(ptr+3*x+0) = 0;
      *(ptr+3*x+1) = 0;
//      *(ptr+3*x+2) = 0;
    }
  }
}

void CoutImage(IplImage* img)
{
  for(int y=0;y<img->height;y++)
  {
    uchar* ptr = (uchar*)(img->imageData + y*img->widthStep);
    for(int x=0;x<img->width;x++)
    {
      std::cout<<int(ptr[3*x + 0])<<"\\"<<int(ptr[3*x + 1])<<"\\"<<int(ptr[3*x + 2])<<"\t";
    }
    std::cout<<std::endl;
  }
}

void chapter3()
{
  //opertate cvmat struct
//    CvMat* rotmat;
//    rotmat = cvCreateMat(3,3,CV_32FC1);
//    cvZero(rotmat);
//    float vals[] = {0.0,1.1,2.2,3.3,4.4,5.5};
//    float element_1_1 = 100.5;
  //  *((float*)cvPtr2D(rotmat,1,1,NULL)+rotmat->step/4) = element_1_1;
//    float* ptr = (float*)(rotmat->data.ptr);
//    *ptr = element_1_1;
  //  float i = CV_MAT_ELEM(*rotmat,float,1,1);
//    double i = cvGetReal2D(rotmat,0,0);
//    std::cout<<"rotmat = "<<(*rotmat).cols<<"\t"<<(*rotmat).rows
//            <<"\t"<<i<<std::endl;
//  cvReleaseMat(rotmat);

  //load image
  IplImage* img = cvLoadImage("Lena.png",CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
  IplImage* img2 = cvCloneImage(img);

  //output image information
    std::cout<<"nsize = "<<img->nSize<<"\t"<<"width = "<<img->width<<"\t"<<"height = "<<img->height<<"\t"
             <<"ID = "<<img->ID<<"\t"<<"imageSize = "<<img->imageSize<<"\t"<<"widthstep = "<<img->widthStep<<"\t"
             <<"nChannels = "<<img->nChannels<<"\t"<<"alphaChannels = "<<img->alphaChannel<<"\t"
             <<"depth = "<<img->depth<<"\t"<<"colorModel = "<<img->colorModel<<"\t"<<"channelseq = "<<img->channelSeq<<std::endl;
    std::cout<<"--------------seperate line---------------------"<<std::endl;
//    std::cout<<"img data"<<std::endl;
//    CoutImage(img2);

  //set image elements
  //  MaximumChannel(img2);

  //select interesting area and operate
//  int x_ = 200;
//  int y_ = 200;
//  int width_ = 150;
//  int height_ = 150;
//  int add_ = 300;
//  cvSetImageROI(img2,cvRect(x_,y_,width_,height_));
//  cvAddS(img2,cvScalar(add_),img2);
//  cvResetImageROI(img2);

  //select interesting area using widthstep,build a new image struct header to organize datas
//  IplImage *sub_img = cvCreateImageHeader(cvSize(width_,height_),img2->depth,img2->nChannels);
//  sub_img->origin = img2->origin;
//  sub_img->widthStep = img2->widthStep;
//  sub_img->imageData = img2->imageData + y_*img2->widthStep + x_*img2->nChannels;
//  cvAddS(sub_img,cvScalar(add_),sub_img);
//  cvReleaseImageHeader(&sub_img);


  //save datas to disk
//  CvFileStorage* fs = cvOpenFileStorage("cfg.xml",0,CV_STORAGE_WRITE);
//  cvWriteInt(fs,"frame_count",10);
//  cvStartWriteStruct(fs,"frame_size",CV_NODE_SEQ);
//  cvWriteInt(fs,NULL,320);
//  cvWriteInt(fs,NULL,200);
//  cvEndWriteStruct(fs);
//  cvWrite(fs,"img_matrix",img);
//  cvWrite(fs,"color_cvt_matrix",rotmat);
//  cvReleaseFileStorage(&fs);
//  cvSaveImage("image1.jpg",img);

  //show image
  cvNamedWindow("Example1",CV_WINDOW_AUTOSIZE);
  cvShowImage("Example1",img);
  cvNamedWindow("Example2",CV_WINDOW_AUTOSIZE);
  cvShowImage("Example2",img2);
  cvMoveWindow("Example2",600,0);
  cvWaitKey(0);
  cvReleaseImage(&img);
  cvReleaseImage(&img2);
  cvDestroyWindow("Example1");
  cvDestroyWindow("Example2");
}

//CvRect box;
//bool drawing_box = false;

//void draw_box(IplImage* img,CvRect rect){
//  cvRectangle(
//    img,
//    cvPoint(box.x,box.y),
//    cvPoint(box.x+box.width,box.y+box.height),
//    cvScalar(0x00,0x00,0xff));
//}

//void my_mouse_callback(int event,int x,int y,int flags,void* param)
//{
//  IplImage* image = (IplImage*) param;
//  switch (event) {
//  case CV_EVENT_MOUSEMOVE:{
//    if(drawing_box){
//      box.width = x-box.x;
//      box.height = y-box.y;
//    }
//  }
//  break;
//  case CV_EVENT_LBUTTONDOWN:{
//    drawing_box = true;
//    box = cvRect(x,y,0,0);
//  }
//  break;
//  case CV_EVENT_LBUTTONUP:{
//    drawing_box = false;
//    if(box.width<0){
//      box.x+=box.width;
//      box.width *=-1;
//    }
//    if(box.height<0){
//      box.y+=box.height;
//      box.height*=-1;
//    }
////    draw_box(image,box);
//  }
//  break;
//  }
//}

//void MouseCall()
//{
//  box = cvRect(-1,-1,0,0);
//  IplImage* image = cvCreateImage(cvSize(600,600),IPL_DEPTH_8U,3);
//  cvZero(image);
//  IplImage* temp = cvCloneImage(image);
//  cvNamedWindow("Box Example");
//  cvSetMouseCallback("Box Example",my_mouse_callback,(void*) image);
//  while(1){
//    cvCopy(image,temp);
//    if(drawing_box) draw_box(temp,box);
//    cvShowImage("Box Example",temp);
//    if(cvWaitKey(15)==27) break;
//  }
//  cvReleaseImage(&image);
//  cvReleaseImage(&temp);
//  cvDestroyWindow("Box Example");
//}

//int g_switch_value = 0;

void switch_callback(int position){
  if(position == 0){
//    switch_off_function();
  }else{
//    switch_on_function();
  }
}

//void TrackBarTest()
//{
//  cvNamedWindow("Demo window",1);
//  cvCreateTrackbar("Switch","Demo window",&g_switch_value,1,switch_callback);
//  while(1){
//    if(cvWaitKey(15)==27)break;
//  }
//}

void LoadVideo()
{//  cvSetMouseCallback("source",my_mouse_callback,NULL);


  //  for(int i=0;i<img_gray->height;i++)
  //  {
  //    uchar* ptr=(uchar*)(img_gray->imageData+i*img_gray->widthStep);
  //    for(int j=0;j<img_gray->width;j++)Lena
  //      ptr[j] = 254;
  //  }

  //  //hist test
  //  Mat r_plane(img.rows,img.cols,CV_8UC1)
  CvCapture* capture = cvCreateCameraCapture(0);
  if(capture==NULL)
    std::cout << "Camera capture failed!"<<std::endl;
  CvVideoWriter* writer = cvCreateVideoWriter("computer_camera.flv",CV_FOURCC('F', 'L', 'V', '1'),30,cvSize(848,480));
  if(writer==NULL)
    std::cout << "video save structure generate failed!"<<std::endl;
  double width_ = cvGetCaptureProperty(capture,3);
  double height_ = cvGetCaptureProperty(capture,4);
  double fps_ = cvGetCaptureProperty(capture,5);
  double fourcc_ = cvGetCaptureProperty(capture,6);
  std::cout<<"video width = "<<width_<<"\t"<<"height = "<<height_<<std::endl;
  std::cout<<"video fps = "<<fps_<<"\t"<<"fourcc = "<<fourcc_<<std::endl;
//  VideoCapture capture(0);
//  while (1)
//  {
//      Mat frame;
//      capture >> frame;
//      imshow("读取视频", frame);
//      waitKey(30);
//  }
//  return 0;
  cvNamedWindow("1", CV_WINDOW_AUTOSIZE);
  IplImage* img = NULL;
  while (1)
  {
  img = cvQueryFrame(capture);
  if (!img) break;
  IplImage* temp_ = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
  cvConvertImage(img,temp_,CV_CVTIMG_FLIP);
  cvShowImage("1", temp_);
  if (27 == cvWaitKey(33))
  break;

  cvWriteFrame(writer,img);
//  cvWaitKey(0);
  }
  cvReleaseCapture(&capture);
  cvReleaseVideoWriter(&writer);
  cvDestroyWindow("1");
}

int num_p = 0;
int x_int = 0;
int y_int = 0;
vector<int> x_temp(4);
vector<int> y_temp(4);
bool capture_ = false;

void my_mouse_callback(int event,int x,int y,int flags,void* param)
{
  switch (event) {
  case CV_EVENT_LBUTTONDOWN:{
    capture_ = true;
    x_int = x;
    y_int = y;
    cout<<"x_int = "<<x_int<<"\t"<<"y_int = "<<y_int<<endl;
    cout<<"capture_ = "<<capture_<<endl;
  }
  break;
  case CV_EVENT_LBUTTONUP:{
    if(capture_)
    {
      capture_ = false;
      cout<<"x_int = "<<x_int<<"\t"<<"y_int = "<<y_int<<endl;
      x_temp.at(num_p) = x_int;
      y_temp.at(num_p) = y_int;
      num_p++;
      cout<<"num_p="<<num_p<<"\t"<<"capture_ = "<<capture_<<endl;
      cout<<"x_[num_p] = "<<x_temp.at(num_p-1)<<"\t"<<"y_[num_p] = "<<y_temp.at(num_p-1)<<endl;

    }
  }
  break;
  }
}


void HoughTrans()
{
  cvNamedWindow("source", CV_WINDOW_AUTOSIZE);
//  cvNamedWindow("gray", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("proceed1", CV_WINDOW_AUTOSIZE);
//  cvNamedWindow("proceed2", CV_WINDOW_AUTOSIZE);
  IplImage* img = cvLoadImage("Lena.png",CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
//  IplImage* img = cvLoadImage("2.bmp",CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
//  cout<<"img depth = "<<img->depth<<endl<<"img channels = "<<img->nChannels<<endl;
  IplImage* img_gray = cvCreateImage(cvGetSize(img),8,1);
  cvCvtColor(img,img_gray,CV_BGR2GRAY);
//  cvSetMouseCallback("proceed",my_mouse_callback,NULL);
//  for(int i=0;i<img_gray->height;i++)
//  {
//    uchar* ptr=(uchar*)(img_gray->imageData+i*img_gray->widthStep);
//    for(int j=0;j<img_gray->width;j++)
//      ptr[j] = 254;
//  }
  IplImage* img_copy2 = cvCreateImage(cvGetSize(img),IPL_DEPTH_16S,3);
  IplImage* img_copy = cvCreateImage(cvGetSize(img),img->depth,3);
  IplImage* img_copy_gray = cvCreateImage(cvGetSize(img),img->depth,1);
  cvCanny(img,img_copy_gray,100,150,3);
  Mat img_copy_gray_2 = cvarrToMat(img_copy_gray);
  vector<Vec2f> lines;
  Point point1, point2;
  Mat dstImage;
  dstImage = cvarrToMat(img);
  HoughLines(img_copy_gray_2,lines,1,CV_PI/180,150,0,0);
  for(size_t i=0;i<lines.size();i++)
  {
    float rho = lines[i][0];
    float theta = lines[i][1];
    double a = cos(theta),b = sin(theta);
    double x0 = rho * cos(theta),y0 = rho * sin(theta);
    Point pt1,pt2;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*a);
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*a);
    line(dstImage,pt1,pt2,Scalar(0,0,255),1,CV_AA);
  }
}




int main(int argc,char** argv)
{
  cvNamedWindow("source", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("source2", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("proceed1", CV_WINDOW_AUTOSIZE);
//  cvNamedWindow("proceed2", CV_WINDOW_AUTOSIZE);
//  cvNamedWindow("matchSource", CV_WINDOW_AUTOSIZE);
//  cvNamedWindow("match", CV_WINDOW_AUTOSIZE);
//  cvNamedWindow("proceed2", CV_WINDOW_AUTOSIZE);
  clock_t a=clock();
  cout<<"a = "<<a<<endl;
  Mat img = imread("/home/zb/BoZhan/OpenCV/Opencv_tutorials/devel/lib/primary/piece_std2.jpg",CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
  Mat img2 = imread("/home/zb/BoZhan/OpenCV/Opencv_tutorials/devel/lib/primary/11.bmp",CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
//  cvSetMouseCallback("source",my_mouse_callback,NULL);

//  cvtColor(img,img,CV_BGR2GRAY);
  cout<<"img.depth = "<<img.depth()<<"\t"<<"img.channels = "<<img.channels()<<endl;
  cout<<"img.size = "<<img.cols<<" * "<<img.rows<<endl;

  /*corner detect
  vector<Point2f> corners;
  goodFeaturesToTrack(img,corners,500,0.01,10,Mat(),3);
  cout<<"** Number of corners detected: "<<corners.size()<<endl;
  int r = 4;
  for( int i = 0; i < corners.size(); i++ )
     { circle( img, corners[i], r, Scalar(0,0,255), -1, 8, 0 ); }


//  cornerHarris(img,img2,4,3,0.04);
//  normalize( img2, img2, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
//  convertScaleAbs( img2, img2 );
//  int i_ = 0;
//  for( int j = 0; j < img2.rows ; j++ )
//     { for( int i = 0; i < img2.cols; i++ )
//          {
//            if( (int) img2.at<float>(j,i) > 10 )
//              {
//               circle( img, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
//               i_++;
//              }
//          }
//     }
//  cout<<"i = "<<i_<<endl;
  */

  //houghlines
//  Mat img_copy_gray(img.size(),img.depth(),1);
//  Canny(img,img_copy_gray,100,150,3);
//  vector<Vec2f> lines;
//  Point point1, point2;
//  Mat dstImage;
//  HoughLines(img_copy_gray,lines,1,CV_PI/180,530,0,0);
//  for(size_t i=0;i<lines.size();i++)
//  {
//    float rho = lines[i][0];
//    float theta = lines[i][1];
//    double a = cos(theta),b = sin(theta);
//    double x0 = rho * cos(theta),y0 = rho * sin(theta);
//    Point pt1,pt2;
//    pt1.x = cvRound(x0 + 1000*(-b));
//    pt1.y = cvRound(y0 + 1000*a);
//    pt2.x = cvRound(x0 - 1000*(-b));
//    pt2.y = cvRound(y0 - 1000*a);
//    line(img,pt1,pt2,Scalar(0,0,255),1,CV_AA);
//  }

  //Features detect and match

//  Point2f center(img2.cols,img2.rows);
//  int Height = img2.rows;
//  int Width = img2.cols;
//  int maxLength = int(sqrt(double(Height*Height + Width*Width)));
//  Mat rot = getRotationMatrix2D(center,10,0.8);
//  warpAffine(img2,img2,rot,Size(Width,Height));
  vector<KeyPoint> key1,key2;
//  Ptr<SIFT> sift;
//  sift = SIFT::create();
  Ptr<SURF> surf;
  surf=SURF::create(800);
//  BFMatcher matcher;
  FlannBasedMatcher matcher;
  Mat c,d;
  vector<DMatch> matches;
  surf->detectAndCompute(img,Mat(),key1,c);
  surf->detectAndCompute(img2,Mat(),key2,d);
  drawKeypoints(img,key1,img,Scalar(255,0,0));
  drawKeypoints(img2,key2,img2,Scalar(255,0,0));
  matcher.match(c,d,matches);
  sort(matches.begin(),matches.end());
  vector<DMatch> good_matches;
  int ptsPairs = min(500,(int)(matches.size()*0.15));
  cout<<ptsPairs<<endl;
  for(int i=0;i<ptsPairs;i++)
  {
    good_matches.push_back(matches[i]);
  }
  Mat outimg;
  drawMatches(img,key1,img2,key2,good_matches,outimg,Scalar(255,0,0),Scalar(0,0,0),vector<char>());

  vector<Point2f> obj;
  vector<Point2f> scene;
  for(size_t i =0;i<good_matches.size();i++)
  {
    obj.push_back(key1[good_matches[i].queryIdx].pt);
    scene.push_back(key2[good_matches[i].trainIdx].pt);
  }

  vector<Point2f> scene_corners(5);
  vector<Point2f> obj_corners(5);
  obj_corners[0] = Point( 0 , 0 );
  obj_corners[1] = Point( img.cols , 0 );
  obj_corners[2] = Point( img.cols , img.rows );
  obj_corners[3] = Point( 0 , img.rows );
  obj_corners[4] = Point(100,464);
  Mat H = findHomography(obj,scene,RANSAC);
  perspectiveTransform(obj_corners,scene_corners,H);
  line( outimg , scene_corners[0] + Point2f( ( float ) img.cols , 0 ) , scene_corners[1] + Point2f( ( float ) img.cols , 0 ) , Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
  line( outimg , scene_corners[1] + Point2f( ( float ) img.cols , 0 ) , scene_corners[2] + Point2f( ( float ) img.cols , 0 ) , Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
  line( outimg , scene_corners[2] + Point2f( ( float ) img.cols , 0 ) , scene_corners[3] + Point2f( ( float ) img.cols , 0 ) , Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
  line( outimg , scene_corners[3] + Point2f( ( float ) img.cols , 0 ) , scene_corners[0] + Point2f( ( float ) img.cols , 0 ) , Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
  Point2f center;
  center.x= scene_corners[4].x;
  center.y= scene_corners[4].y;
  cout<<"target_x = "<<center.x<<"\t"<<"target_y = "<<center.y<<endl;
  circle(img2,center,5,Scalar(0,0,255),5);
  double angle;
  if((scene_corners[3].y - scene_corners[0].y)!=0)
  {
    angle = atan((scene_corners[3].x - scene_corners[0].x)/(scene_corners[3].y - scene_corners[0].y))*180/PI;
    cout<<"angle = "<<angle<<endl;
  }
  else
  {
     cout<<"angle = "<<0<<endl;
  }

  clock_t b = clock();
  cout<<"b = "<<b<<endl;
  cout<<"CLOCKS_PER_SEC = "<<CLOCKS_PER_SEC<<endl;
  cout<<"time = "<<((double)(b - a))/CLOCKS_PER_SEC<<"s"<<endl;



  //OPtical flow
  /*
  const int MAX_CORNERS = 500;
  VideoCapture cap(0);
  if( !cap.isOpened() )
  {
      printf("can not open video file\n");
      return -1;
  }
  Mat img1,img2;
  cap>>img1;
  cvtColor(img1,img1,CV_BGR2GRAY);
  waitKey(3000);
  cap>>img2;
  cvtColor(img2,img2,CV_BGR2GRAY);
  CvSize img_sz = img1.size();
  int win_size = 10;
  Mat img3 = img1;
  Mat img2_copy = img2;
  int corner_count = MAX_CORNERS;
  vector<Point2f> cornersA(MAX_CORNERS);
  goodFeaturesToTrack(
  img1,//the input image
  cornersA,//temp image whose result is meaningful
  corner_count,//the maximum number of points
  0.01,
  5.0
  );
  cout<<"corner_count = "<<corner_count<<endl;
  cornerSubPix(img1,cornersA,cvSize(win_size,win_size),cvSize(-1,-1),cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));
  vector<unsigned char> features_found(MAX_CORNERS);
  vector<float> feature_errors(MAX_CORNERS);
  vector<Point2f> cornersB(MAX_CORNERS);
  calcOpticalFlowPyrLK(
        img1,
        img2,
        cornersA,
        cornersB,
        features_found,
        feature_errors,
        cvSize(win_size,win_size),
        5,cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ),0);
  for( int i=0; i<corner_count; i++ )
  {
     if( features_found[i]==0|| feature_errors[i]>550 )
     {
         printf("Error is %f/n",feature_errors[i]);
         continue;
      }
      printf("Got it/n");
      CvPoint p0 = cvPoint(cvRound( cornersA[i].x ),cvRound( cornersA[i].y ));
      CvPoint p1 = cvPoint(cvRound( cornersB[i].x ),cvRound( cornersB[i].y ));
      line( img3, p0, p1, CV_RGB(255,0,0),2 );
  }*/


//  cvtColor(img,img,CV_BGR2HSV);
//  cvtColor(img2,img2,CV_BGR2HSV);
//  IplImage* img = cvLoadImage("2.bmp",CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_ANYCOLOR);
//  cout<<"img depth = "<<img.depth()<<endl<<"img channels = "<<img.channels()<<endl;


//  Mat img_gray(img.rows,img.cols,CV_8UC1);
//  Mat img_bin(img.rows,img.cols,CV_8UC1);
//  Mat img_bin_copy(img.rows,img.cols,CV_8UC1);
//  cvtColor(img,img_gray,CV_BGR2GRAY);
    /*get contours//////////////////
  Mat edges;
  Canny(img_gray,edges,10,120);
  threshold(edges,img_bin,30,255,CV_THRESH_BINARY);
  img_bin.copyTo(img_bin_copy);
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(img_bin,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_NONE);
  Mat resultImage = Mat::zeros(img.size(),CV_8U);
  for(int i=0;i<contours.size();i++)
  {
     drawContours(resultImage,contours,i,Scalar(255,0,255),2,8,hierarchy);
  }

  vector<vector<Point>> contours_poly(contours.size());
  vector<Rect> boundRect(contours.size());
  vector<Point2f> center(contours.size());
  vector<float> radius(contours.size());
  vector<RotatedRect> box(contours.size());
  vector<Moments> mu(contours.size() );
  vector<Point2f> mc( contours.size() );
  vector<vector<Point>> hull(contours.size());

  for(int i=0;i<contours.size();i++)
  {
    approxPolyDP(Mat(contours[i]),contours_poly[i],3,true);
    boundRect[i] = boundingRect(Mat(contours[i]));
    minEnclosingCircle(Mat(contours[i]),center[i],radius[i]);
//    box[i] = fitEllipse(Mat(contours[i]));
    rectangle(img_bin_copy,boundRect[i],Scalar(255,0,0),2,8,0);
    circle(img_bin_copy,center[i],radius[i],Scalar(255,0,0),2,8,0);
    mu[i] = moments( contours[i], false );
    mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
//    circle( img, mc[i], 4, Scalar(255,0,0), -1, 8, 0 );

    convexHull(Mat(contours[i]),hull[i],false);
    drawContours(img,hull,i,Scalar(255,0,0),1,8,vector<Vec4i>(),0,Point());
//    ellipse(img_bin_copy, box[i].center, box[i].size*0.5f, box[i].angle, 0, 360, Scalar(0,255,255), 1, LINE_AA);
  }
  ///////////////////////*/
//  Mat markers;
//  markers = Mat::zeros(img.size(),CV_8UC3);
//  watershed(img,markers);
//  pyrMeanShiftFiltering(img,markers,20,40,2);









  //temple match
//  cvSetMouseCallback("source",my_mouse_callback,NULL);


//  for(int i=0;i<img_gray->height;i++)
//  {
//    uchar* ptr=(uchar*)(img_gray->imageData+i*img_gray->widthStep);
//    for(int j=0;j<img_gray->width;j++)Lena
//      ptr[j] = 254;
//  }

//  //hist test
//  Mat r_plane(img.rows,img.cols,CV_8UC1);
//  Mat g_plane(img.rows,img.cols,CV_8UC1);
//  Mat b_plane(img.rows,img.cols,CV_8UC1);
//  vector<Mat> channels_t;
//  split(img,channels_t);
//  r_plane = channels_t.at(0);
//  g_plane = channels_t.at(1);
//  b_plane = channels_t.at(2);
//  Mat Planes[] = {r_plane,g_plane};

//  Mat img_copy(img.rows,img.cols,CV_8UC3);
//  Mat img_copy2(img.rows,img.cols,CV_8UC3);
//  Mat img_copy_gray(img.rows,img.cols,CV_8UC1);
//  Mat img_dist_tran(img.rows,img.cols,CV_32FC1);

//  const int bins = 256;
//  const int channels[2]={0,1};
//  const int histSize[2]={bins,bins};
//  float hranges[2]={0,255};
//  const float* ranges[2]={hranges,hranges};
//  MatND hist,hist2;
//  calcHist(&img,1,channels,Mat(),hist,2,histSize,ranges);
//  calcHist(&img2,1,channels,Mat(),hist2,2,histSize,ranges);
//  double maxVal = 0;
//  double minVal = 0;
//  double maxVal2 = 0;
//  double minVal2 = 0;
//  minMaxLoc(hist,&minVal,&maxVal,0,0);
//  minMaxLoc(hist2,&minVal2,&maxVal2,0,0);
//  cout<<"maxVal = "<<maxVal<<"\t"<<"minVal = "<<minVal<<endl;
//  cout<<"maxVal2 = "<<maxVal2<<"\t"<<"minVal2 = "<<minVal2<<endl;
//  int scale = 1;
//  Mat histimg(bins*scale,bins*scale,CV_8UC3,Scalar(255,255,255));
//  Mat histimg2(bins*scale,bins*scale,CV_8UC3,Scalar(255,255,255));
////  for(int h =0;h<histSize2;h++)
////  {
////    float binVal=hist.at<float>(h);
////    int intensity=static_cast<int>(binVal*hpt/maxVal);
////    line(histimg,Point(h,histSize2),Point(h,histSize2-intensity),Scalar::all(0));
////  }
//  for(int h=0;h<bins;h++)
//  {
//    for(int s=0;s<bins;s++)
//    {
//      float bin_val = hist.at<float>(h,s);
//      int intensity = cvRound(bin_val*255/maxVal);
//      rectangle(histimg,Point(h*scale,s*scale),Point((h+1)*scale-1,(s+1)*scale-1),CV_RGB(intensity,intensity,intensity));
//      float bin_val2 = hist2.at<float>(h,s);
//      int intensity2 = cvRound(bin_val2*255/maxVal2);
//      rectangle(histimg2,Point(h*scale,s*scale),Point((h+1)*scale-1,(s+1)*scale-1),CV_RGB(intensity2,intensity2,intensity2));

//    }
//  }
//  double divesity = compareHist(hist,hist2,CV_COMP_CORREL);
//  cout<<"hist diversity = "<<divesity<<endl;





  /*distance transform*/
//  Canny(img_gray,img_copy_gray,100,150,3);
//  threshold(img_copy_gray,img_copy_gray,100,255,CV_THRESH_BINARY_INV);
//  distanceTransform(img_copy_gray,img_dist_tran,CV_DIST_L2,3,CV_32F);
//  normalize(img_dist_tran,img_dist_tran,0,1,CV_MINMAX);



//  CvPoint2D32f srcTri[3],dstTri[3];
//  CvMat* rot_mat = cvCreateMat(2,3,CV_32FC1);
//  CvMat* warp_mat = cvCreateMat(2,3,CV_32FC1);
//  img_copy->origin = img->origin;
//  srcTri[0].x = 0;
//  srcTri[0].y = 0;
//  srcTri[1].x = img->width-1;
//  srcTri[1].y = 0;
//  srcTri[2].x = 0;
//  srcTri[2].y = img->height-1;
//  dstTri[0].x = img->width*0.0;
//  dstTri[0].y = img->height*0.33;
//  dstTri[1].x = img->width*0.85;
//  dstTri[1].y = img->width*0.25;
//  dstTri[2].x = img->width*0.15;
//  dstTri[2].y = img->width*0.7;
//  cvGetAffineTransform(srcTri,dstTri,warp_mat);
//  cvWarpAffine(img,img_copy,warp_mat);
//  CvPoint2D32f center = CvPoint2D32f(img->width/2,img->height/2);
//  double angle = -50.0;
//  double scale = 0.6;
//  cv2DRotationMatrix(center,angle,scale,rot_mat);
//  cvWarpAffine(img,img_copy2,rot_mat);

//  cvLogPolar(img,img_copy,cvPoint2D32f(img->width/2,img->height/2),10.0);
//  cvLogPolar(img_copy,img_copy2,cvPoint2D32f(img->width/2,img->height/2),10.0,CV_INTER_LINEAR|CV_WARP_INVERSE_MAP);

  /**********************************fft************************************************
  int M = getOptimalDFTSize(img_gray.rows);
  int N = getOptimalDFTSize(img_gray.cols);
  Mat padded;
  copyMakeBorder(img_gray,padded,0,M-img_gray.rows,0,N-img_gray.cols,BORDER_CONSTANT,Scalar::all(0));
  Mat planes[]={Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F)};
  Mat complexImg;
  merge(planes,2,complexImg);
  dft(complexImg,complexImg);
  split(complexImg,planes);
  magnitude(planes[0],planes[1],planes[0]);
  Mat mag = planes[0];
  mag += Scalar::all(1);
  log(mag,mag);
  mag = mag(Rect(0,0,mag.cols&-2,mag.rows&(-2)));
  Mat _magI = mag.clone();
  normalize(_magI,_magI,0,1,CV_MINMAX);

  int cx = mag.cols/2;
  int cy = mag.rows/2;
  Mat tmp;
  Mat q0(mag,Rect(0,0,cx,cy));
  Mat q1(mag,Rect(cx,0,cx,cy));
  Mat q2(mag,Rect(0,cy,cx,cy));
  Mat q3(mag,Rect(cx,cy,cx,cy));
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
  normalize(mag,mag,0,1,CV_MINMAX);
  **********************************************************************************/


   // @match temple
/*
  cvSetMouseCallback("source",my_mouse_callback,NULL);
  imshow("source",img);
  Mat img_copy;
  Mat result;
  int flag = 0;
  img.copyTo(img_copy);
  while(1){
//    if(capture_)   cvFloodFill(img,cvPoint(x_,y_),cvScalar(100,100,100),cvScalar(10,10,10),cvScalar(10,10,10),NULL,8,NULL);
    if(((abs(x_end-x_int)>=1)&&(abs(y_end-y_int)>=1))&&(flag ==0 ))
    {
      if((x_end>0)&&(y_end>0))
      {
        img_copy.copyTo(img);
        Mat temp_(abs(x_end-x_int),abs(y_end-y_int),CV_8UC3);
        Rect roi_rect = Rect(x_int,y_int,abs(x_end-x_int),abs(y_end-y_int));
        img(roi_rect).copyTo(temp_);
        imwrite("temp.jpg",temp_);
        rectangle(img,roi_rect,cvScalar(255,0,0),1);
        imshow("source",img);
        imshow("proceed1",temp_);

        int result_cols = img2.cols - temp_.cols + 1;
        int result_rows = img2.rows - temp_.rows + 1;
        result.create(result_cols,result_rows,CV_32FC1);
        matchTemplate(img2,temp_,result,CV_TM_SQDIFF_NORMED);
        normalize(result,result,0,1,NORM_MINMAX,-1,Mat());

        double minVal; double maxVal; Point minLoc; Point maxLoc;
        Point matchLoc;
        minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
        matchLoc = minLoc;
        rectangle( img2, matchLoc, Point( matchLoc.x + temp_.cols , matchLoc.y + temp_.rows ), Scalar::all(0), 2, 8, 0 );
        rectangle( result, matchLoc, Point( matchLoc.x + temp_.cols , matchLoc.y + temp_.rows ), Scalar::all(0), 2, 8, 0 );
        flag = 1;
      }
    }
//    imshow("proceed1",temp_);
//    imshow("backproj",backproj);
//    imshow("proceed2",mag);
    if(flag==1)
    {
      imshow( "source", img );
      imshow("proceed1",img2);
      imshow( "proceed2", result );
    }
    if(char(cvWaitKey(15))==27)break;
  }
  waitKey(0);
  */


  //get piece_std
//  Mat temp_;
//  double angle;
//  while(1)
//  {
//    imshow( "source", img );
////    imshow( "source2", img_copy );
////    imshow( "proceed1", img_copy_gray);
////    imshow("proceed2",outimg);
//    if(char(cvWaitKey(15))==27)break;
//  }

//  if(num_p==1)
//  {
//    vector<Point2f> obj_corners(4);
//    obj_corners[0] = Point( 321-100, 261 );
//    obj_corners[1] = Point( x_temp[1], y_temp[1] );
//    obj_corners[2] = Point( x_temp[2], y_temp[2] );
//    obj_corners[3] = Point( x_temp[3], y_temp[3] );
////    imwrite("temp.jpg",temp_);
//    line( img , obj_corners[0], obj_corners[1], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
//    line( img , obj_corners[1], obj_corners[2], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
//    line( img , obj_corners[2], obj_corners[3], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
//    line( img , obj_corners[3], obj_corners[0], Scalar( 0 , 255 , 0 ) , 2 , LINE_AA );
//    if((obj_corners[3].y - obj_corners[0].y)!=0)
//    {
//    angle = atan((obj_corners[3].x - obj_corners[0].x)/(obj_corners[3].y - obj_corners[0].y));
//    cout<<"angle = "<<angle<<endl;
//    }
//    else
//    {
//      cout<<"the rotation of input image is not correct!"<<endl;
//    }
//    Point rotaCent;
//    rotaCent.y = obj_corners[3].x;
//    rotaCent.x = obj_corners[3].y;
//    Mat img_std = img.clone();
//    Mat rotaMat = getRotationMatrix2D(rotaCent, -angle*180/PI, 1);
//    warpAffine(img, img, rotaMat, Size(img.cols, img.rows));
//    Rect roi_rect = Rect(obj_corners[0].x,obj_corners[0].y,210+100,464+100);
//    img(roi_rect).copyTo(temp_);
//    imwrite("piece_std2.jpg",temp_);

//  }

  while(1)
  {
    imshow( "source", img );
    imshow( "source2", img2 );
    imshow( "proceed1", outimg);
//    imshow("proceed2",outimg);
    if(char(cvWaitKey(15))==27)break;
  }
//  cvReleaseImage(&img_gray);
  cvDestroyAllWindows();
}





//int main()
//{
//    VideoCapture cap(0);
//    if( !cap.isOpened() )
//    {
//        printf("can not open video file\n");
//        return -1;
//    }

//    namedWindow("image", WINDOW_NORMAL);
//    namedWindow("foreground mask", WINDOW_NORMAL);


//    BackgroundSubtractorCodeBook bgcbModel;

//    Mat inputImage,outputMaskCodebook;

//    for(int i=0;;++i)
//    {
//        cap>>inputImage;
//        if( inputImage.empty() )
//            break;
//        if(i==0)
//        {
//            bgcbModel.initialize(inputImage,outputMaskCodebook);

//        }
//        else if (i<=30&&i>0)
//        {
//            bgcbModel.updateCodeBook(inputImage);
//            if (i==30)
//            {
//                bgcbModel.clearStaleEntries();
//            }
//        }
//        else
//        {
//            bgcbModel.backgroudDiff(inputImage,outputMaskCodebook);
//        }
//        imshow("image",inputImage);
//        imshow("foreground mask",outputMaskCodebook);

//        int c = waitKey(30);
//        if (c == 'q' || c == 'Q' || (c & 255) == 27)
//            break;
//    }

//    return 0;
//}


