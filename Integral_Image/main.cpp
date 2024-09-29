//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

//options

struct ArgumentList {
  std::string image_name;                   //!< image file name
  int wait_t;                               //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv);

void Integral_image(const cv::Mat& src, cv::Mat& out);


int main(int argc, char **argv)
{
  int frame_number = 0;
  char frame_name[256];
  bool exit_loop = false;
  int imreadflags = cv::IMREAD_COLOR; 

  std::cout<<"Integral Image"<<std::endl;

  //////////////////////
  //parse argument list:
  //////////////////////
  ArgumentList args;
  if(!ParseInputs(args, argc, argv)) {
    exit(0);
  }

  while(!exit_loop)
  {
    //generating file name
    //
    //multi frame case
    if(args.image_name.find('%') != std::string::npos)
      sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
    else //single frame case
      sprintf(frame_name,"%s",args.image_name.c_str());

    //opening file
    std::cout<<"Opening "<<frame_name<<std::endl;

    cv::Mat image = cv::imread(frame_name, imreadflags);
    if(image.empty())
    {
      std::cout<<"Unable to open "<<frame_name<<std::endl;
      return 1;
    }

    std::cout << "The image has " << image.channels() << 
      " channels, the size is " << image.rows << "x" << image.cols << " pixels " <<
      " the type is " << image.type() <<
      " the pixel size is " << image.elemSize() <<
      " and each channel is " << image.elemSize1() << (image.elemSize1()>1?" bytes":" byte") << std::endl;

    //////////////////////
    //processing code here
    cv::Mat gray(image.size(), CV_8UC1);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    std::cout<< "Grayscale image first row:"<< std::endl;
    for(int i=0; i<gray.cols; i++){
      std::cout<< int(gray.at<uchar>(0,i))<< " ";
    }
    std::cout<< std::endl; 
    
    cv::Mat ii;
    Integral_image(gray, ii);
    std::cout<< ii.size()<< std::endl;
    std::cout<< "Integral image first row:"<< std::endl;
    for(int i=0; i<ii.cols; i++){
      std::cout << ii.at<int>(0,i)<< " ";
    }
    std::cout<< std::endl;
    
    /////////////////////

    //display image
    cv::namedWindow("original image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE if you want the window to adapt to image size
    cv::imshow("original image", image);
    cv::namedWindow("Greyscale", cv::WINDOW_NORMAL);
    cv::imshow("Greyscale", gray);

    //wait for key or timeout
    unsigned char key = cv::waitKey(args.wait_t);
    std::cout<<"key "<<int(key)<<std::endl;

    //here you can implement some looping logic using key value:
    // - pause
    // - stop
    // - step back
    // - step forward
    // - loop on the same frame

    switch(key)
    {
      case 'p':
	std::cout << "Mat = "<< std::endl << image << std::endl;
	break;
      case 'q':
	exit_loop = 1;
	break;
      case 'c':
	std::cout << "SET COLOR imread()" << std::endl;
	imreadflags = cv::IMREAD_COLOR;
	break;
      case 'g':
	std::cout << "SET GREY  imread()" << std::endl;
	imreadflags = cv::IMREAD_GRAYSCALE; // Y = 0.299 R + 0.587 G + 0.114 B
	break;
    }

    frame_number++;
  }

  return 0;
}


bool ParseInputs(ArgumentList& args, int argc, char **argv);

#if NATIVE_OPTS
// cumbersome, it requires to use "=" for args, i.e. -i=../images/Lenna.pgm
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  args.wait_t=0;

  cv::CommandLineParser parser(argc, argv,
      "{input   i|../images/Lenna.png|input image, Use %0xd format for multiple images.}"
      "{wait    t|0                  |wait before next frame (ms)}"
      "{help    h|<none>             |produce help message}"
      );

  if(parser.has("help"))
  {
    parser.printMessage();
    return false;
  }

  args.image_name = parser.get<std::string>("input");
  args.wait_t     = parser.get<int>("wait");

  return true;
}

#else

#include <unistd.h>
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  int c;
  args.wait_t = 0;

  while ((c = getopt (argc, argv, "hi:t:")) != -1)
    switch (c)
    {
      case 't':
        args.wait_t = atoi(optarg);
        break;
      case 'i':
        args.image_name = optarg;
        break;
      case 'h':
      default:
        std::cout<<"usage: " << argv[0] << " -i <image_name>"<<std::endl;
        std::cout<<"exit:  type q"<<std::endl<<std::endl;
        std::cout<<"Allowed options:"<<std::endl<<
          "   -h                       produce help message"<<std::endl<<
          "   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
          "   -t arg                   wait before next frame (ms)"<<std::endl<<std::endl;
        return false;
    }
  return true;
}

void Integral_image(const cv::Mat& src, cv::Mat& out){

  out.create(src.size(), CV_32SC1);

  for(int r=0; r<out.rows; r++){
    for(int c=0; c<out.cols; c++){
      int current_value = src.at<uchar>(r,c);
      int psx = 0, pup = 0, pdi = 0;
      if(r-1 >= 0){
        pup = out.at<int>(r-1,c);
      }
      if(c-1 >= 0){
        psx = out.at<int>(r,c-1);
      }
      if(r-1 >= 0 && c-1 >= 0){
        pdi = out.at<int>(r-1,c-1);
      }
      
      out.at<int>(r,c) = current_value + psx + pup + pdi;

    }
  }
}

#endif




