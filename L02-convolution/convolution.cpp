//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

void addPadding(const cv::Mat image, cv::Mat &out, int vp, int hp); 

void myfilter2D(const cv::Mat& src, const cv::Mat& krn, cv::Mat& out,  int stride=1);

void contBright(const cv::Mat&src, const cv::Mat& out, float a=1, int b=0);


struct ArgumentList {
  std::string image_name;		    //!< image file name
  int wait_t;                     //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv);

int main(int argc, char **argv)
{
  int frame_number = 0;
  char frame_name[256];
  bool exit_loop = false;
  int ksize = 3;
  int stride = 1;
  float gain = 1; //contrast
  int bias = 0; //brightness

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

    cv::Mat image = cv::imread(frame_name);
    if(image.empty())
    {
      std::cout<<"Unable to open "<<frame_name<<std::endl;
      return 1;
    }

    //display image
    cv::namedWindow("original image", cv::WINDOW_NORMAL);
    cv::imshow("original image", image);


    // PROCESSING

    // convert to grey scale for following processings
    cv::Mat grey;

    cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    cv::namedWindow("grey", cv::WINDOW_NORMAL);
    cv::imshow("grey", grey);

    /*Changing contrast and brightness*/
    cv::Mat result(grey.rows,grey.cols,grey.type());
    contBright(grey,result,gain,bias);


    cv::namedWindow("ContBright", cv::WINDOW_NORMAL);
    cv::imshow("ContBright", result);

    cv::Mat blurred;

    // BOX FILTERING
    // void cv::boxFilter(InputArray src, OutputArray dst, int ddepth, Size ksize, Point anchor = Point(-1,-1), bool normalize = true, int borderType = BORDER_DEFAULT )
    cv::boxFilter(grey, blurred, CV_8U, cv::Size(ksize, ksize));
    cv::namedWindow("Box filter Smoothing", cv::WINDOW_NORMAL);
    cv::imshow("Box filter Smoothing", blurred);
    std::cout <<"Box ok"<< std::endl;

    cv::Mat custom_kernel(ksize, ksize, CV_32FC1, 1.0/(ksize*ksize));
    // also possible as cv::Mat custom_kernel = 1.0/(ksize*ksize) * cv::Mat::ones(ksize, ksize, CV_32FC1);
    cv::Mat custom_blurred, custom_display;
    //cv::filter2D(grey, custom_blurred, CV_32F, custom_kernel);
    myfilter2D(grey, custom_kernel, custom_blurred, stride);
    cv::convertScaleAbs(custom_blurred, custom_display);
    cv::namedWindow("Box filter Smoothing (custom)", cv::WINDOW_NORMAL);
    cv::imshow("Box filter Smoothing (custom)", custom_display);
    blurred.copyTo(grey);

    // SOBEL FILTERING
    // void cv::Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)

    cv::Mat g, gx, gy, agx, agy, ag;
    cv::Mat my_gx, my_gy;
    cv::Mat h_sobel = (cv::Mat_<float>(3,3) << 1, 0, -1,
      2, 0, -2,
      1, 0, -1);
    cv::Mat v_sobel = h_sobel.t();
    myfilter2D(grey, h_sobel, my_gx, stride); //calcolo derivata orizzontale
    myfilter2D(grey, v_sobel, my_gy, stride); //calcolo derivata verticale
    my_gx.convertTo(gx, CV_32FC1);
    my_gy.convertTo(gy, CV_32FC1);
    //cv::Sobel(grey, gx, CV_32F, 1, 0, 3);
    //cv::Sobel(grey, gy, CV_32F, 0, 1, 3);
    // compute magnitude
    cv::pow(gx.mul(gx) + gy.mul(gy), 0.5, g);
    // compute orientation
    cv::Mat orientation(gx.size(), CV_32FC1); 
    float *dest = (float *)orientation.data;
    float *srcx = (float *)gx.data;
    float *srcy = (float *)gy.data;
    float *magn = (float *)g.data;
    for(int i=0; i<gx.rows*gx.cols; ++i)
      dest[i] = magn[i]>50 ? atan2f(srcy[i], srcx[i]) + 2*CV_PI: 0;
    // scale on 0-255 range
    cv::convertScaleAbs(gx, agx);
    cv::convertScaleAbs(gy, agy);
    cv::convertScaleAbs(g, ag);
    cv::namedWindow("sobel verticale", cv::WINDOW_NORMAL);
    cv::imshow("sobel verticale", agx);
    cv::namedWindow("sobel orizzontale", cv::WINDOW_NORMAL);
    cv::imshow("sobel orizzontale", agy);
    cv::namedWindow("sobel magnitude", cv::WINDOW_NORMAL);
    cv::imshow("sobel magnitude", ag);    

    // trick to display orientation
    cv::Mat adjMap;
    cv::convertScaleAbs(orientation, adjMap, 255 / (2*CV_PI));
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);
    cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
    cv::imshow("sobel orientation", falseColorsMap);    



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
      case 's':
	if(stride != 1)
	  --stride;
	std::cout << "Stride: " << stride << std::endl;
	break;
      case 'S':
	++stride;
	std::cout << "Stride: " << stride << std::endl;
	break;
      case 'C':
  gain+=0.1;
  std::cout<<"Contrast: "<<gain<<std::endl;
  break;
      case 'c':
  if(gain>=0.1){
    gain-=0.1;
  }
  std::cout<<"Contrast: "<<gain<<std::endl;
  break;
      case 'B':
  bias+=2;
  std::cout<<"Brightness: "<<bias<<std::endl;
  break;
      case 'b':
  if(bias!=0){
    bias-=2;
  }
  std::cout<<"Brightness: "<<bias<<std::endl;
  break;
      case 'd':
	cv::destroyAllWindows();
	break;
      case 'p':
	std::cout << "Mat = "<< std::endl << image << std::endl;
	break;
      case 'k':
	{
	  static int sindex=0;
	  int values[]={3, 5, 7, 11 ,13};
	  ksize = values[++sindex%5];
	  std::cout << "Setting Kernel size to: " << ksize << std::endl;
	}
	break;
      case 'g':
	break;
      case 'q':
	exit(0);
	break;
    }

    frame_number++;
  }

  return 0;
}

#if 0
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  args.wait_t=0;

  cv::CommandLineParser parser(argc, argv,
      "{input   i|in.png|input image, Use %0xd format for multiple images.}"
      "{wait    t|0     |wait before next frame (ms)}"
      "{help    h|<none>|produce help message}"
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

  while ((c = getopt (argc, argv, "hi:t:c:b:")) != -1)
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

void addPadding(const cv::Mat image, cv::Mat &out, int vp, int hp){
  out = cv::Mat(image.rows+2*vp,image.cols+2*hp,image.type(), cv::Scalar(0));
  for(int r=vp;r<out.rows-vp;r++){
    for(int c=hp;c<out.cols-hp;c++){
      for(int k=0;k<out.channels();k++){
        out.data[((r*out.cols+c)*out.elemSize()+k*out.elemSize1())]=image.data[((r-vp)*image.cols+(c-hp))*image.elemSize()+k*image.elemSize1()];
      }
    }
  }
}
void myfilter2D(const cv::Mat& src, const cv::Mat& krn, cv::Mat& out,  int stride){
  /*calcolo le dimensioni dell'immagine di uscita*/
  int  outsizeh = (src.rows+ (krn.rows/2)*2 - krn.rows)/(float)stride + 1;
  int  outsizew = (src.cols+ (krn.cols/2)*2 - krn.cols)/(float)stride + 1;
  out = cv::Mat(outsizeh,outsizew,CV_32SC1);

  /*effettuo il padding sull'immagine originale*/
  cv::Mat image;
  addPadding(src,image,krn.rows/2,krn.cols/2);

  int xc=krn.cols/2;  //colonna pixel centrale del kernel
  int xr=krn.rows/2;  //riga pixel centrale del kernel
  int *outbuffer = (int *) out.data;
  float *kernel = (float *) krn.data;

  for(int i=0; i<out.rows; i++){
    for(int j=0; j<out.cols;j++){
      /*coordinate pixel centrale del kernel durante gli spostamenti*/
      int origr = i*stride + xr;
      int origc = j*stride + xc;

      float sum=0;

      for(int ki=-xr; ki<=xr; ki++){
        for(int kj=-xc; kj<=xc; kj++){
          sum += image.data[(origr+ki)*image.cols + (origc+kj)]*kernel[(ki+xr)*krn.cols + (kj+xc)];
        }
      }
      outbuffer[i*out.cols+j] = sum;
    }
  }
}

void contBright(const cv::Mat&src, const cv::Mat& out, float a, int b){

  for(int y=0; y<out.rows; y++){
    for(int x=0; x<out.cols; x++){
      for(int k=0; k<out.channels(); k++){
        out.data[(y*out.cols+x)*out.elemSize()+k*out.elemSize1()]= a*src.data[(y*src.cols+x)*src.elemSize()+k*src.elemSize1()]+b;
      }
    }
  }
}

#endif


