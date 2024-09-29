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
  /*type 0 --> previous
    type 1 --> running avg
    type 2 --> exponential running avg*/
  int type;
  int th = 50; //treshold for background subtraction
  int k = 10;  //number of images used to compute the background 
  float alpha = 0.5;  //parameter for exp. run. avg.
};

bool ParseInputs(ArgumentList& args, int argc, char **argv);

void ComputeForeground(const cv::Mat& b, const cv::Mat& i, const cv::Mat& out, int t);


/*Per eseguire la sequenza: simple -i Candela_m1.10_%06d.pgm -t 50*/

int main(int argc, char **argv)
{
  int frame_number = 0;
  char frame_name[256];
  bool exit_loop = false;
  int imreadflags = cv::IMREAD_COLOR; 
  std::vector<cv::Mat> vect;

  std::cout<<"Simple program."<<std::endl;
  //////////////////////
  //parse argument list:
  //////////////////////
  ArgumentList args;
  if(!ParseInputs(args, argc, argv)) {
    exit(0);
  }

  cv::Mat prev_bg;

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

    cv::Mat image = cv::imread(frame_name, CV_8UC1); //imreadflags
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
    cv::Mat start(image.rows,image.cols,image.type(),cv::Scalar(0));
    cv::Mat out(image.rows,image.cols,image.type());
    
    
    
    
    if(args.type == 0){
      /*Previous Image*/
      
      if(vect.size()>1){
        ComputeForeground(vect[frame_number-2],vect[frame_number-1],out,args.th);
        cv::namedWindow("background-Previous Image", cv::WINDOW_NORMAL);
        cv::imshow("background-Previous Image", vect[frame_number-1]);
        cv::namedWindow("foreground", cv::WINDOW_NORMAL);
        cv::imshow("foreground", out);

      }

    }
    if(args.type == 1){
      /*Running Average*/
      if(!vect.empty()){
        cv::Mat back(image.rows,image.cols,CV_32FC1, cv::Scalar(0)), fg;
        for(int i=0; i<vect.size(); i++){
          cv::accumulate(vect[i],back);
        }
        back /= vect.size();
        back.convertTo(back, CV_16SC1);
        image.convertTo(fg, CV_16SC1);
        cv::threshold(cv::abs(fg-back), fg, args.th, 255, cv::THRESH_BINARY);

        back.convertTo(back, CV_8UC1);
        fg.convertTo(fg, CV_8UC1);
        cv::namedWindow("background-Running Average", cv::WINDOW_NORMAL);
        cv::imshow("background-Running Average", back);
        cv::namedWindow("foreground", cv::WINDOW_NORMAL);
        cv::imshow("foreground", fg);
      }

    }

    if(args.type == 2){
      /*Exponential Running Average*/
      if(prev_bg.empty()){
        image.copyTo(prev_bg);
      }else{
        cv::Mat fg, bg;
        
        image.convertTo(fg, CV_16SC1);
        prev_bg.convertTo(bg, CV_16SC1);

        cv::threshold(cv::abs(fg-bg), fg, args.th, 255, cv::THRESH_BINARY);
        fg.convertTo(fg, CV_8UC1);
        cv::namedWindow("background-Exp. Running Average", cv::WINDOW_NORMAL);
        cv::imshow("background-Exp. Running Average", prev_bg);
        cv::namedWindow("foreground", cv::WINDOW_NORMAL);
        cv::imshow("foreground", fg);

        prev_bg = args.alpha*prev_bg + (1-args.alpha)*image;

      }

    }
    
    vect.push_back(image);
    if(args.type == 1){
      if(vect.size()>args.k){
        vect.erase(vect.begin());
      }
    }
    
    std::cout<<"Sisze: "<<vect.size()<<std::endl;
    /////////////////////

    //display image
    cv::namedWindow("original image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE if you want the window to adapt to image size
    cv::imshow("original image", image);


    

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
  args.type = 0;  //default Prvious image

  while ((c = getopt (argc, argv, "hi:t:b:s:k:a:")) != -1)
    switch (c)
    {
      case 't':
        args.wait_t = atoi(optarg);
        break;
      case 'i':
        args.image_name = optarg;
        break;
      case 'h':
      case 'b':
        args.type = atoi(optarg);
        break;
      case 's':
        args.th = atoi(optarg);
        break;
      case 'k':
        args.k = atoi(optarg);
        break;
      case 'a':
        args.alpha = atoi(optarg);
        break;
      default:
        std::cout<<"usage: " << argv[0] << " -i <image_name>"<<std::endl;
        std::cout<<"exit:  type q"<<std::endl<<std::endl;
        std::cout<<"Allowed options:"<<std::endl<<
          "   -h                       produce help message"<<std::endl<<
          "   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
          "   -b arg                   type of BS (0-->Prev., 1-->Run.AVG., 2-->Exp.Run.AVG.)"<<std::endl<<
          "   -s arg                   treshold value for BS"<<std::endl<<
          "   -k arg                   number of frame used for background average"<<std::endl<<
          "   -a arg                   alpha (EXP. Run. AVG.)"<<std::endl<<
          "   -t arg                   wait before next frame (ms)"<<std::endl<<std::endl;
        return false;
    }
  return true;
}

void ComputeForeground(const cv::Mat& b, const cv::Mat& i, const cv::Mat& out, int t){
  cv::Mat bg, f,o;

  b.convertTo(bg,CV_16SC1);
  i.convertTo(f,CV_16SC1);

  cv::threshold(cv::abs(f-bg), o, t, 255, cv::THRESH_BINARY);
  o.convertTo(out, CV_8UC1);

  /*for(int y=0; y<out.rows; y++){
    for(int x=0; x<out.rows; x++){
      for(int k=0; k<out.channels(); k++){
        int diff = i.data[(y*i.cols+x)*i.elemSize()+k*i.elemSize1()]-b.data[(y*b.cols+x)*b.elemSize()+k*b.elemSize1()];
        int val = abs(diff);
        //std::cout<<"!!!!DIFF: "<<val<<" !!!!"<<std::endl;
        if(val > t){
          out.data[(y*out.cols+x)*out.elemSize()+k*out.elemSize1()] = 255;
        }else{
          out.data[(y*out.cols+x)*out.elemSize()+k*out.elemSize1()] = 0;
        }
      }
    }
  }*/

}


#endif


