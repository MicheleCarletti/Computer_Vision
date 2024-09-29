//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <bits/stdc++.h>


void binarize(const cv::Mat& in,  const cv::Mat&out, unsigned int th);

void erosion(const cv::Mat& in, const cv::Mat& se, const cv::Point2i& origin, cv::Mat& out);

void dilation(const cv::Mat& in, const cv::Mat& se, const cv::Point2i& origin, cv::Mat& out);

void labeling(const cv::Mat& in, cv::Mat& out);

struct ArgumentList {
  std::string image_name;		    //!< image file name
  int wait_t;                     //!< waiting time
  int th;                       //threshold for binary
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

    /*create binary*/
    cv::Mat bn(image.size(),CV_8UC1);
    binarize(grey,bn,args.th);

    cv::namedWindow("Binary", cv::WINDOW_NORMAL);
    cv::imshow("Binary", bn);

    cv::Mat SE(3,3,CV_8UC1,cv::Scalar(255));

    cv::Mat i;
    bn.copyTo(i);

    cv::Mat resultE;
    cv::Mat resultD;
    cv::Point2i center(1,1);
    cv::Mat e,d;
    cv::Mat aperture;

    erosion(i,SE,center,resultE);
    //cv::erode(bn,e,SE,center);

    cv::namedWindow("Eroded Image", cv::WINDOW_NORMAL);
    cv::imshow("Eroded Image", resultE);
    /*cv::namedWindow("Erode", cv::WINDOW_NORMAL);
    cv::imshow("Erode", e);*/


    dilation(i,SE,center,resultD);
    //cv::dilate(bn,d,SE,center);

    cv::namedWindow("Dilated Image", cv::WINDOW_NORMAL);
    cv::imshow("Dilated Image", resultD);
    /*cv::namedWindow("Dilate", cv::WINDOW_NORMAL);
    cv::imshow("Dilate", d);*/

    dilation(resultE, SE, center, aperture);
    cv::namedWindow("Opening", cv::WINDOW_NORMAL);
    cv::imshow("Opening", aperture);

    cv::Mat l;
    labeling(aperture,l);

    /*color map for labels*/
    cv::Mat adjMap;
    l.convertTo(adjMap,CV_8UC1);
    cv::Mat falseColorMap;
    cv::applyColorMap(adjMap,falseColorMap,cv::COLORMAP_RAINBOW);

    cv::namedWindow("Labeled Image", cv::WINDOW_NORMAL);
    cv::imshow("Labeled Image",falseColorMap);



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
  args.th = 100;

  while ((c = getopt (argc, argv, "hi:t:c:b:")) != -1)
    switch (c)
    {
      case 't':
	args.wait_t = atoi(optarg);
	break;
      case 'i':
	args.image_name = optarg;
	break;
      case 'b':
  args.th = atoi(optarg);
  break;
      case 'h':
      default:
	std::cout<<"usage: " << argv[0] << " -i <image_name>"<<std::endl;
	std::cout<<"exit:  type q"<<std::endl<<std::endl;
	std::cout<<"Allowed options:"<<std::endl<<
	  "   -h                       produce help message"<<std::endl<<
	  "   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
    "   -b arg                   threhold for binary image"<<std::endl<<
	  "   -t arg                   wait before next frame (ms)"<<std::endl<<std::endl;
	return false;
    }
  return true;
}


void binarize(const cv::Mat& in,  const cv::Mat&out, unsigned int th){  
  cv::threshold(in, out, th, 255, cv::THRESH_OTSU); //cv::THRESH_BINARY binary using th, cv::THRESH_OTSU apply Otsu's method to compute threshold (ignore th)
  out.convertTo(out, CV_8UC1);
}

inline bool _morph_set(unsigned char a)
{
  return a;
}

inline void _morph_set255(unsigned char &x)
{
  x = 255;
}


inline bool _morph_notset(unsigned char a)
{
  return !a;
}

inline void _morph_set0(unsigned char &x)
{
  x = 0;
}
void morph(const cv::Mat &in, const cv::Mat &se, const cv::Point2i &origin, cv::Mat &out, bool (* check)(unsigned char), void (*set)(unsigned char &))
{
  // check image type
  if(in.type() != CV_8UC1)
  {
    std::cerr << "Error: " << __func__ << "() wrong image type as input" << std::endl;
    exit(EXIT_FAILURE);
  }

  // initially output can be considered as copy of input
  in.copyTo(out);

  // wrt convolution, dealing with padding is difficult, since we have to consider both shape and origin of SE
  // for border pixels we consider partial overlay

  // iterate on all output/input pixels
  for(int r = 0; r < in.rows; ++r){
    for(int c = 0; c < in.cols; ++c){
      // iterate on all se "pixels"
      bool ok = false;
      for(int kr = 0; kr < se.rows && !ok; ++kr){
        for(int kc = 0; kc < se.cols; ++kc){
          unsigned char se_value = se.data[kc + se.cols*kr];
          if(!se_value)
            continue;  // optimization trick
	        // compute image coords where se point is superimposed
	        int origy = r + (kr - origin.y);
	        int origx = c + (kc - origin.x);

	        // check image boundaries
	        if(origy >= 0 and origy < in.rows and origx >= 0 and origx < in.cols)
          {
            if(check(in.data[origx + origy*in.cols])) // at least one SE element is superimposed on a '1' pixel image
            {
              set(out.data[c + r*out.cols]);
              ok = true;
              break;
            }
          }
        }
      }
    }
  }
}

void erosion(const cv::Mat& in, const cv::Mat& se, const cv::Point2i& origin, cv::Mat& out){
  morph(in, se, origin, out, _morph_notset, _morph_set0);
  
}
void dilation(const cv::Mat& in, const cv::Mat& se, const cv::Point2i& origin, cv::Mat& out){
  morph(in,se,origin,out,_morph_set,_morph_set255);

}

void labeling(const cv::Mat& in, cv::Mat& out){
  cv::Mat labels = cv::Mat::zeros(in.rows,in.cols,CV_16UC1);
  uint16_t* labels_data = (uint16_t*) labels.data;
  uint16_t label = 5; //initial label

  /*First pass*/
  labels_data[0]=label;
  /*first row*/
  for(int c=1; c<in.cols; c++){
    if(in.data[c] == in.data[c-1]){
      labels_data[c]=labels_data[c-1];
    }else{
      labels_data[c]=label++;
    }
  } 

  /*other rows*/
  std::map <uint16_t,uint16_t> same_labels;
  std::map <uint16_t, uint16_t>::iterator it;

  /*first pixel of each row*/
  for(int r=1; r<in.rows; r++){
    if(in.data[r*in.cols]==in.data[(r-1)*in.cols]){ 
       labels_data[r*in.cols]=labels_data[(r-1)*in.cols];
    }else{
       labels_data[r*in.cols]=label++;
    }

    /*other pixels*/
    for(int c=0; c<in.cols; c++){
      uint8_t p = in.data[r*in.cols+c]; //current pixel
      uint8_t pup = in.data[(r-1)*in.cols+c]; //upper pixel
      uint8_t psx = in.data[r*in.cols+c-1]; //left pixel

      uint16_t &plabel = labels_data[r*in.cols+c]; //label of current pixel
      uint16_t puplabel = labels_data[(r-1)*in.cols+c]; //label of upper pixel
      uint16_t psxlabel = labels_data[r*in.cols+c-1]; //label of left pixel

      /*same value and same label*/
      if(p==pup and p==psx and puplabel==psxlabel){
        plabel = puplabel;
      }
      /*same value but different label*/
      else if(p==pup and p==psx and puplabel!=psxlabel){
        plabel=std::min(puplabel,psxlabel);
        same_labels[std::max(puplabel,psxlabel)]=std::min(puplabel,psxlabel); //track the correspondance
      }
      /*same value left pixel only*/
      else if(p==psx and p!=pup){
        plabel=psxlabel;
      }
      /*same value upper pixel only*/
      else if(p==pup and p!=psx){
        plabel=puplabel;
      }
      else{
        plabel=label++;
      }
        
    }
  }

  /*Second pass*/
  /*several passseges are needed to align the labels*/
  bool swapped;
  do{
    swapped=false;
    for(int y=0; y<in.rows; y++){
      for(int x=0; x<in.cols; x++){
        uint16_t &plabel = labels_data[y*in.cols+x];
        while((it=same_labels.find(plabel)) != same_labels.end()){
          swapped=true;
          plabel = it->second;
        }
      }
    }
  }while(swapped);
  out=labels;

}


#endif


