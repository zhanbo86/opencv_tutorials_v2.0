#include "main.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "parameters.h"

using namespace cv;
using namespace std;

struct PT
{
    int x;
    int y;
};
struct LINE
{
    PT pStart;
    PT pEnd;
};
Point CrossPoint(const LINE *line1, const LINE *line2)
{

    Point pt;
    // line's cpmponent ,y = a1*x + b
    int dx1 = line1->pEnd.x-line1->pStart.x;
    int dx2 = line2->pEnd.x-line2->pStart.x;
    double a1,b1,a2,b2;
    if(dx1!=0&&dx2 !=0)
    {
      a1 = ((float)(line1->pEnd.y-line1->pStart.y))/((float)(dx1));
      b1 = ((float)(line1->pStart.y*line1->pEnd.x-line1->pEnd.y*line1->pStart.x))/((float)(dx1));
      a2 = ((float)(line2->pEnd.y-line2->pStart.y))/((float)(dx2));
      b2 = ((float)(line2->pStart.y*line2->pEnd.x-line2->pEnd.y*line2->pStart.x))/((float)(dx2));

      if(abs(a1-a2)>0.5)//two lines are not parrele lines
      {
        pt.x = (int)(((float)(b2-b1))/((float)(a1-a2)));
        pt.y = (int)(((float)(a1*b2-a2*b1))/((float)(a1-a2)));
        return pt;
      }
      else
      {
        Point p_error;
        p_error.x = 0;
        p_error.y = 0;
        return p_error;
      }
    }
    else if(dx1==0&&dx2!=0)
    {
      a2 = ((float)(line2->pEnd.y-line2->pStart.y))/((float)(dx2));
      b2 = ((float)(line2->pStart.y*line2->pEnd.x-line2->pEnd.y*line2->pStart.x))/((float)(dx2));
      pt.x = line1->pStart.x;
      pt.y = a2*pt.x + b2;
      return pt;
    }
    else if(dx1!=0&&dx2==0)
    {
      a1 = ((float)(line1->pEnd.y-line1->pStart.y))/((float)(dx1));
      b1 = ((float)(line1->pStart.y*line1->pEnd.x-line1->pEnd.y*line1->pStart.x))/((float)(dx1));
      pt.x = line2->pStart.x;
      pt.y = a1*pt.x + b1;
      return pt;
    }
    else
    {
      Point p_error;
      p_error.x = 0;
      p_error.y = 0;
      return p_error;
    }

}


int find_longest(vector<Vec4i> Lines)
{
  float max_length = 0;
  float line_length = 0;
  int max_index = 0;
  for(int i=0;i<Lines.size();i++)
  {
    line_length = sqrt(((double)(Lines[i][1] - Lines[i][3]))*((double)(Lines[i][1] - Lines[i][3]))+
                             ((double)(Lines[i][0] - Lines[i][2]))*((double)(Lines[i][0] - Lines[i][2])));
    if(line_length>max_length)
    {
      max_length = line_length;
      max_index = i;
    }
  }
  return max_index;
}

int main()
{
  ///***********************************set parameters*************************************///
  float plate_width_pixels = ((float)(plate_width))/((float)(cam_field_width))*
                             ((float)(cam_pixel_width));
  float plate_height_pixels = ((float)(plate_height))/((float)(cam_field_height))*
                              ((float)(cam_pixel_height));
  float min_plate_pixels = min(plate_width_pixels,plate_height_pixels);
  float close_thread = min_plate_pixels*0.15;
  float open_thread = min_plate_pixels*0.01;
  float side_length = min_plate_pixels;
  double canny_1_thresh = 100;
  double canny_2_thresh = 200;
  double min_distance_hori = plate_height_pixels*0.5;
  double min_distance_vert = plate_width_pixels*0.5;




  ///***********************************process image*************************************///
  Mat image = imread("/home/zb/BoZhan/OpenCV/BMT0602/2.bmp", IMREAD_GRAYSCALE);
	if (!image.data)
		return -1;
	Mat img_filter;
	blur(image, img_filter, Size(3, 3));//先使用3*3内核来降噪
	//GaussianBlur(image, img_filter, Size(3, 3), 0, 0);
	//bilateralFilter(image, img_filter, 25, 25 * 2, 25 / 2);
	//img_filter = image;
	
	std::cout << "image  = " << image.cols << " * " << image.rows << " / " << "channels = " << image.channels() << std::endl;
	namedWindow("source", 0);
	imshow("source", img_filter);
	waitKey(0);
	cvDestroyWindow("source");

	Mat RGBImg;
	cvtColor(img_filter, RGBImg, CV_GRAY2BGR);

//  Mat src_thresh;
//  threshold(img_filter,src_thresh,0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY);
//  namedWindow("src_thresh", 0);
//  imshow("src_thresh", src_thresh);
//  waitKey(0);
//  cvDestroyWindow("src_thresh");

  Mat img_canny_1;
  Canny(img_filter, img_canny_1, canny_1_thresh, canny_1_thresh*3,3);
  namedWindow("canny", 0);
  imshow("canny", img_canny_1);
  waitKey(0);
  cvDestroyWindow("canny");

  //remove isolated points
  vector<vector<Point>> canny_contours;
  vector<Vec4i> canny_hierarcy;
  double canny_contour_length;
  findContours(img_canny_1, canny_contours, canny_hierarcy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  for (int i = 0; i<canny_contours.size(); i++)
  {
    canny_contour_length = arcLength(canny_contours[i], true);
    if(canny_contour_length<30)
    {
      Rect rect = boundingRect(canny_contours[i]);
      rectangle(img_canny_1,cvPoint(rect.x,rect.y),cvPoint(rect.x+rect.width ,rect.y+rect.height),CV_RGB(0,0,0),-1,CV_AA,0);
    }
  }
  Mat canny_thresh;
  threshold(img_canny_1,canny_thresh,0, 255, CV_THRESH_BINARY);
  namedWindow("canny_thresh", 0);
  imshow("canny_thresh", canny_thresh);
  waitKey(0);
  cvDestroyWindow("canny_thresh");



  //find contours points
  Mat img_contours(canny_thresh.rows,canny_thresh.cols,CV_8UC1,Scalar(0, 0, 0));
  uchar* p_canny = canny_thresh.ptr<uchar>(0);
  uchar* p_img = img_contours.ptr<uchar>(0);
  for(int i=0;i<canny_thresh.rows;i++)
  {
    p_canny = canny_thresh.ptr<uchar>(i);
    p_img = img_contours.ptr<uchar>(i);
    for(int j=0;j<canny_thresh.cols;j++)
    {
      if(p_canny[j]==255)
      {
        p_img[j]=255;
        break;
      }
    }
  }
  for(int i=0;i<canny_thresh.rows;i++)
  {
    p_canny = canny_thresh.ptr<uchar>(i);
    p_img = img_contours.ptr<uchar>(i);
    for(int j=canny_thresh.cols-1;j>=0;j--)
    {
      if(p_canny[j]==255)
      {
        p_img[j]=255;
        break;
      }
    }
  }
  p_canny = canny_thresh.ptr<uchar>(0);
  p_img = img_contours.ptr<uchar>(0);
  for(int i=0;i<canny_thresh.cols;i++)
  {
    for(int j=0;j<canny_thresh.rows;j++)
    {
      if(p_canny[i+j*canny_thresh.cols]==255)
      {
        p_img[i+j*canny_thresh.cols] = 255;
        break;
      }
    }
  }
  for(int i=0;i<canny_thresh.cols;i++)
  {
    for(int j=canny_thresh.rows-1;j>=0;j--)
    {
      if(p_canny[i+j*canny_thresh.cols]==255)
      {
        p_img[i+j*canny_thresh.cols] = 255;
        break;
      }
    }
  }
  namedWindow("img_contours", 0);
  imshow("img_contours", img_contours);
  waitKey(0);
  cvDestroyWindow("img_contours");


////  cv::Mat element_close(200,200,CV_8U,cv::Scalar(1));
//  cv::Mat element_close(close_thread,close_thread,CV_8U,cv::Scalar(1));
//  cv::Mat closed;
//  cv::morphologyEx(canny_thresh, closed, cv::MORPH_CLOSE, element_close);
//  namedWindow("closed", 0);
//  imshow("closed", closed);
//  waitKey(0);
//  cvDestroyWindow("closed");

////  cv::Mat element_open(10,10,CV_8U,cv::Scalar(1));
//  cv::Mat element_open(open_thread,open_thread,CV_8U,cv::Scalar(1));
//  cv::Mat closeopened;
//  cv::morphologyEx(closed, closeopened, cv::MORPH_OPEN, element_open);
//  namedWindow("closeopened", 0);
//  imshow("closeopened", closeopened);
//  waitKey(0);
//  cvDestroyWindow("closeopened");


//  Mat img_canny_2;
//  Canny(closeopened, img_canny_2, canny_2_thresh, canny_2_thresh*1.5,3);
//  namedWindow("canny", 0);
//  imshow("canny", img_canny_2);
//  waitKey(0);
//  cvDestroyWindow("canny");


//  //find contours
//  vector<vector<Point>> contours;
//  vector<Point> contour_max;
//  vector<Vec4i> hierarcy;
//  findContours(closeopened, contours, hierarcy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//  cout << "num=" << contours.size() << endl;

//  float contour_length_max=0;
//  float contour_length=0;
//  int contour_num_max=0;
//  for (int i = 0; i<contours.size(); i++)
//  {
//    contour_length = arcLength(contours[i], true);
//    if(contour_length>contour_length_max)
//    {
//      contour_length_max = contour_length;
//      contour_num_max = i;
//    }
//  }
//  printf("contous_num_max = %d\n",contour_num_max);
//  contour_max = contours[contour_num_max];
//  Mat contours_result = RGBImg.clone();
//  drawContours(contours_result, contours, contour_num_max, Scalar(0, 0, 255), 2, 8);
//  namedWindow("contours", 0);
//  imshow("contours", contours_result);
//  waitKey(0);
//  cvDestroyWindow("contours");


  ///***********************************find four lines*************************************///
  vector<Vec4i> Lines;
  HoughLinesP(img_contours, Lines, 1, CV_PI / 360, side_length*0.2, 0.1*side_length, 0.5*side_length);
  printf("lines size = %d\n",Lines.size());


//  /*画出all直线*/
//  Mat lines_result = RGBImg.clone();
//  for(int i=0;i<Lines.size();i++)
//  {
//      line(lines_result, Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]), Scalar(0, 255, 0), 2, 8);
//  }
//  namedWindow("lines", 0);
//  imshow("lines", lines_result);
//  waitKey(0);
//  cvDestroyWindow("lines");

  //divide lines to two verticle lines and horizontal lines
  vector<Vec4i> Lines_verti,Lines_hori;
  Lines_verti.clear();
  Lines_hori.clear();
  for(int i=0;i<Lines.size();i++)
  {
      if(abs((Lines[i][0] - Lines[i][2]))<3)
      {
        Lines_verti.push_back(Lines[i]);
      }
      else
      {
        double ki = abs((double)(Lines[i][1] - Lines[i][3]) / (double)(Lines[i][0] - Lines[i][2]));
        if(ki<0.5)
        {
          Lines_hori.push_back(Lines[i]);
        }
        else
        {
          Lines_verti.push_back(Lines[i]);
        }
      }
  }

  //find longest line in horizontal
  int max_hori_index = find_longest(Lines_hori);
//  printf("max_hori_index = %d\n",max_hori_index);

  //divide horizontal lines to two kinds
  Vec4i line_hori_top,line_hori_bottom;
  vector<Vec4i> Lines_hori_sep;
  int pos_flag_bottom=2;//represent max_hori_index line position
  double distance_hori;
  for(int i=0;i<Lines_hori.size();i++)
  {
    if(i !=max_hori_index)
    {
      if(abs((Lines_hori[max_hori_index][2]-Lines_hori[max_hori_index][0]))>3)
      {
        distance_hori = ((double)(Lines_hori[max_hori_index][3]-Lines_hori[max_hori_index][1]))*((double)(Lines_hori[i][0]-Lines_hori[max_hori_index][0]))/((double)(Lines_hori[max_hori_index][2]-Lines_hori[max_hori_index][0]))+
                   (double)(Lines_hori[max_hori_index][1] - Lines_hori[i][1]);
      }
      else
      {
        distance_hori = (double)(Lines_hori[max_hori_index][1] - Lines_hori[i][1]);
      }
//      printf("distance_hori is %f\n",distance_hori);

      if(distance_hori>min_distance_hori)
      {
        pos_flag_bottom = 1;
        Lines_hori_sep.push_back(Lines_hori[i]);
        line_hori_bottom = Lines_hori[max_hori_index];
//        printf("flag is 1\n");

      }
      else if(distance_hori<=-min_distance_hori)
      {
        pos_flag_bottom = 0;
        Lines_hori_sep.push_back(Lines_hori[i]);
        line_hori_top = Lines_hori[max_hori_index];
//        printf("flag is 0\n");
      }
    }
  }

  //find longest line in another horizontal
  int second_max_hori_index = find_longest(Lines_hori_sep);
  if(pos_flag_bottom==1)
  {
    line_hori_top = Lines_hori_sep[second_max_hori_index];
  }
  else if(pos_flag_bottom==0) {
    line_hori_bottom = Lines_hori_sep[second_max_hori_index];
  }

  //find longest line in verticle
  int max_vert_index = find_longest(Lines_verti);
//  printf("max_vert_index = %d\n",max_vert_index);

  //divide verticle lines to two kinds
  Vec4i line_vert_left,line_vert_right;
  vector<Vec4i> Lines_vert_sep;
  int pos_flag_left=2;//represent max_vert_index line positon
  double distance_vert;
  for(int i=0;i<Lines_verti.size();i++)
  {
    if(i !=max_vert_index)
    {
      if(abs((Lines_verti[max_vert_index][1]-Lines_verti[max_vert_index][3]))>3)
      {
        distance_vert = ((double)(Lines_verti[i][1]-Lines_verti[max_vert_index][3]))*((double)(Lines_verti[max_vert_index][0]-Lines_verti[max_vert_index][2]))/((double)(Lines_verti[max_vert_index][1]-Lines_verti[max_vert_index][3]))+
                   (double)(Lines_verti[max_vert_index][2] - Lines_verti[i][0]);
      }
      else
      {
        distance_vert = (double)(Lines_verti[max_vert_index][0] - Lines_verti[i][0]);
      }

      if(distance_vert>min_distance_vert)//right
      {
        pos_flag_left = 0;
        Lines_vert_sep.push_back(Lines_verti[i]);
        line_vert_right = Lines_verti[max_vert_index];

      }
      else if(distance_vert<=-min_distance_vert)//left
      {
        pos_flag_left = 1;
        Lines_vert_sep.push_back(Lines_verti[i]);
        line_vert_left = Lines_verti[max_vert_index];
      }
    }
  }

  //find longest line in another verticle
  int second_max_vert_index = find_longest(Lines_vert_sep);
  if(pos_flag_left==1)
  {
    line_vert_right = Lines_vert_sep[second_max_vert_index];
  }
  else if(pos_flag_left==0) {
    line_vert_left = Lines_vert_sep[second_max_vert_index];
  }


  Mat four_lines_result = RGBImg.clone();
  line(four_lines_result, Point(line_hori_top[0], line_hori_top[1]), Point(line_hori_top[2], line_hori_top[3]), Scalar(0, 255, 0), 2, 8);
  line(four_lines_result, Point(line_hori_bottom[0], line_hori_bottom[1]), Point(line_hori_bottom[2], line_hori_bottom[3]), Scalar(0, 255, 0), 2, 8);
  line(four_lines_result, Point(line_vert_left[0], line_vert_left[1]), Point(line_vert_left[2], line_vert_left[3]), Scalar(0, 255, 0), 2, 8);
  line(four_lines_result, Point(line_vert_right[0], line_vert_right[1]), Point(line_vert_right[2], line_vert_right[3]), Scalar(0, 255, 0), 2, 8);
  namedWindow("lines", 0);
  imshow("lines", four_lines_result);
  waitKey(0);


  //************************************compute lines cross points*************************************//
  LINE lineTop, lineBottom,lineLeft,lineRight;
  lineTop.pStart.x = line_hori_top[0];
  lineTop.pStart.y = line_hori_top[1];
  lineTop.pEnd.x = line_hori_top[2];
  lineTop.pEnd.y = line_hori_top[3];

  lineBottom.pStart.x = line_hori_bottom[0];
  lineBottom.pStart.y = line_hori_bottom[1];
  lineBottom.pEnd.x = line_hori_bottom[2];
  lineBottom.pEnd.y = line_hori_bottom[3];

  lineLeft.pStart.x = line_vert_left[0];
  lineLeft.pStart.y = line_vert_left[1];
  lineLeft.pEnd.x = line_vert_left[2];
  lineLeft.pEnd.y = line_vert_left[3];

  lineRight.pStart.x = line_vert_right[0];
  lineRight.pStart.y = line_vert_right[1];
  lineRight.pEnd.x = line_vert_right[2];
  lineRight.pEnd.y = line_vert_right[3];

  Point cross_P0 = CrossPoint(&lineTop, &lineLeft);
  Point cross_P1 = CrossPoint(&lineTop, &lineRight);
  Point cross_P2 = CrossPoint(&lineBottom, &lineRight);
  Point cross_P3 = CrossPoint(&lineBottom, &lineLeft);

  if((!(cross_P0.x==0&&cross_P0.y==0))&&(!(cross_P1.x==0&&cross_P1.y==0))
     &&(!(cross_P2.x==0&&cross_P2.y==0))&&(!(cross_P3.x==0&&cross_P3.y==0)))
  {
    circle(four_lines_result, cross_P0, 10, Scalar(0, 255, 0),-1);
    circle(four_lines_result, cross_P1, 10, Scalar(0, 255, 0),-1);
    circle(four_lines_result, cross_P2, 10, Scalar(0, 255, 0),-1);
    circle(four_lines_result, cross_P3, 10, Scalar(0, 255, 0),-1);
    imshow("lines", four_lines_result);
    waitKey(0);
  }

  //************************************locate string *****************************************//
 std::vector<Point2f> plate_real;
 std::vector<Point2f> plate_scene;
 Point plate_real_p0 = cvPoint(0,0);
 Point plate_real_p1 = cvPoint(plate_width,0);
 Point plate_real_p2 = cvPoint(plate_width,plate_height);
 Point plate_real_p3 = cvPoint(0,plate_height);
 plate_real.push_back(plate_real_p0);
 plate_real.push_back(plate_real_p1);
 plate_real.push_back(plate_real_p2);
 plate_real.push_back(plate_real_p3);
 plate_scene.push_back(cross_P0);
 plate_scene.push_back(cross_P1);
 plate_scene.push_back(cross_P2);
 plate_scene.push_back(cross_P3);
 Mat H = findHomography( plate_real, plate_scene, CV_RANSAC );

 std::vector<Point2f> plate_string(4);
 plate_string[0] = cvPoint(string_x,string_y);
 plate_string[1] = cvPoint(string_x + string_width, string_y);
 plate_string[2] = cvPoint(string_x + string_width, string_y + string_height );
 plate_string[3] = cvPoint(string_x, string_y + string_height );
 std::vector<Point2f> scene_string(4);
 perspectiveTransform( plate_string, scene_string, H);
 std::vector<Point2f> scene_string_expand(4);
 scene_string_expand[0].x =  scene_string[0].x - string_pos_dx;
 scene_string_expand[0].y =  scene_string[0].y - string_pos_dy;
 scene_string_expand[1].x =  scene_string[1].x + string_pos_dx;
 scene_string_expand[1].y =  scene_string[1].y - string_pos_dy;
 scene_string_expand[2].x =  scene_string[2].x + string_pos_dx;
 scene_string_expand[2].y =  scene_string[2].y + string_pos_dy;
 scene_string_expand[3].x =  scene_string[3].x - string_pos_dx;
 scene_string_expand[3].y =  scene_string[3].y + string_pos_dy;

 line( four_lines_result, scene_string_expand[0], scene_string_expand[1], Scalar(0, 255, 0), 4 );
 line( four_lines_result, scene_string_expand[1], scene_string_expand[2], Scalar( 0, 255, 0), 4 );
 line( four_lines_result, scene_string_expand[2], scene_string_expand[3], Scalar( 0, 255, 0), 4 );
 line( four_lines_result, scene_string_expand[3], scene_string_expand[0], Scalar( 0, 255, 0), 4 );

 imshow("lines", four_lines_result);
 waitKey(0);
 cvDestroyWindow("lines");


}
