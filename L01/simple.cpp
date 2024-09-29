//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

struct ArgumentList {
  std::string image_name = "../images/bayer_GBRG/GBRG.pgm";		    //!< image file name "../images/Lenna.png" "../images/Lindsey.jpg"
  int wait_t=0;                     //!< waiting time
  int x=0;    //x for crop area
  int y=0;  //y for crop area
  int width=100;  //crop area width
  int height=100; //crop area height
  int padding=0;  //dimensione del padding
};

bool ParseInputs(ArgumentList& args, int argc, char **argv);

int main(int argc, char **argv)
{
  int frame_number = 0;
  char frame_name[256];
  bool exit_loop = false;
  int imreadflags = cv::IMREAD_COLOR; 

  std::cout<<"Simple program."<<std::endl;

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
    cv::Mat A(image, cv::Rect(100,100,200,200));
    //single channel access
    cv::Mat B(100, 100, CV_8UC3);
    for (int i=0; i<B.rows*B.cols; i+= B.elemSize()){
      B.data[i] = i;    //B
      B.data[i+B.elemSize1()] = i+1;  //G
      B.data[i+B.elemSize1()+B.elemSize1()] = i+2;  //R
    }
    //row-column-channel (1 byte) access
    cv::Mat M(100, 100, CV_8UC3);
    for(int v=0; v<M.rows; v++){
      for(int u=0; u<M.cols; u++){
        for(int k=0; k<M.channels(); k++){
          M.data[(u+v*M.cols)*M.channels()+k] = u;
        }    
          }
    }
    cv::Mat D (3, 2, CV_8UC3, cv::Scalar(0,0,255));
    std::cout<<"D = "<<D<<std::endl;
    std::cout << "row =  " << D.rows << " col = " << D.cols << " ch = " << D.channels() << std::endl;
    cv::Vec3b px = D.at<cv::Vec3b>(0,0);  //prelevo il valore di un pixel
    std::cout << "B = " << static_cast<int>(px.val[0]) <<std::endl;
    std::cout << "G = " << static_cast<int>(px.val[1]) <<std::endl;
    std::cout << "R = " << static_cast<int>(px.val[2]) <<std::endl;  
    /*
    cv::Mat C(ima1ge.rows, image.cols, CV_8UC3);
    cv::resize(image, C, cv::Size(), 0.5, 1);
    std::cout<< "Anoter image description: "<< std::endl;
    std::cout << "The image has " << C.channels() << 
      " channels, the size is " << C.rows << "x" << C.cols << " pixels " <<
      " the type is " << C.type() <<
      " the pixel size is " << C.elemSize() <<
      " and each channel is " << C.elemSize1() << (C.elemSize1()>1?" bytes":" byte") << std::endl;
    */
    std::vector<cv::Mat> channels;
    cv::split(D,channels);  //Divido i 3 canali di image 
    //Stampo i pixel del ch R
    std::cout << "Channel R" <<std::endl;
    for(int x=0; x<D.rows; x++){
      for(int y=0; y<D.cols; y++){
        std::cout << static_cast<int>(channels[2].at<uchar>(x,y))<<" ";
      }
      std::cout<<std::endl;
    }
    //Stampo i pixel del ch G
    std::cout << "Channel G" <<std::endl;
    for(int x=0; x<D.rows; x++){
      for(int y=0; y<D.cols; y++){
        std::cout << static_cast<int>(channels[1].at<uchar>(x,y))<<" ";
      }
      std::cout<<std::endl;
    }
    //Stampo i pixel del ch B
    std::cout << "Channel B" <<std::endl;
    for(int x=0; x<D.rows; x++){
      for(int y=0; y<D.cols; y++){
        std::cout << static_cast<int>(channels[0].at<uchar>(x,y))<<" ";
      }
      std::cout<<std::endl;
    }
    /*Subsampled image w/2 h/2*/
    int w_n = image.cols/2;
    int h_n = image.rows/2;

    cv::Mat subsIm(h_n, w_n, CV_8UC3);

    for(int x=0; x<subsIm.rows; x++){
      for(int y=0; y<subsIm.cols; y++){
        subsIm.at<cv::Vec3b>(x,y) = image.at<cv::Vec3b>(x*2,y*2); //prendo 1 riga e 1 colonna ogni 2
      }
    }
    /*Sumbsampled image h/2*/
    cv::Mat subsHIm(h_n, image.cols, CV_8UC3);

    for(int r=0; r<subsHIm.rows; r++){
      for(int c=0; c<subsHIm.cols; c++){
        for(int k=0; k<subsHIm.channels(); k++){
          subsHIm.data[(r*subsHIm.cols+c)*subsHIm.elemSize()+k*subsHIm.elemSize1()]=image.data[(r*image.cols*2+c)*image.elemSize()+k*image.elemSize1()];
        }
        
      }
    }
    /*Subsampling w/2*/
    cv::Mat subsWIm(image.rows, w_n, CV_8UC3);

    for(int r=0; r<subsWIm.rows; r++){
      for(int c=0; c<subsWIm.cols; c++){
        for(int k=0; k<subsWIm.channels(); k++){
          subsWIm.data[(r*subsWIm.cols+c)*subsWIm.elemSize()+k*subsHIm.elemSize1()]=image.data[(r*image.cols+c*2)*image.elemSize()+k*image.elemSize1()];
        }
      }
    }
    /*  Alternative version with .at(x,y) method
    for(int x=0; x<subsWIm.rows; x++){
      for(int y=0; y<subsWIm.cols; y++){
        subsWIm.at<cv::Vec3b>(x,y) = image.at<cv::Vec3b>(x,y*2);
      }
    }*/
    /*Flipped image*/
    cv::Mat flipIm(image.rows, image.cols, image.type());

    for(int x=0; x<flipIm.rows; x++){
      for(int y=0; y<flipIm.cols; y++){
        flipIm.at<cv::Vec3b>(x, flipIm.cols-1-y) = image.at<cv::Vec3b>(x,y);
      }
    }
    /*Vertically flipped image*/
    cv::Mat flipVIm(image.rows, image.cols, image.type());

    for(int x=0; x<flipVIm.rows; x++){
      for(int y=0; y<flipVIm.cols; y++){
        flipVIm.at<cv::Vec3b>(flipVIm.rows-1-x, y) = image.at<cv::Vec3b>(x,y);
      }
    }
    /*cropped image from input*/

    if(args.x<0 ||  args.y<0 || args.x+args.width>image.cols || args.y+args.height>image.rows){ //verifica integrità 
      std::cerr<<"L'area di ritaglio è fuori dai limiti dell'immagine"<<std::endl;
      return -1;
    }
    //crea area ritaglio
    cv::Rect roi(args.x, args.y, args.width, args.height);
    cv::Mat croppedIm = image(roi);

    std::cout<<"Subsample Image: r = "<<subsIm.rows<<" c = "<<subsIm.cols<<std::endl;

    /*0 padded image*/
    int new_h = image.rows + 2*args.padding;
    int new_w = image.cols + 2*args.padding;

    cv::Mat paddedIm(new_h, new_w, image.type(), cv::Scalar(0,0,0));

    image.copyTo(paddedIm(cv::Rect(args.padding, args.padding, image.cols, image.rows)));

    /*shuffled image*/

    int partH = image.rows/2;
    int partW = image.cols/2;

    /*creo le 4 sezioni*/
    cv::Mat part1 = image(cv::Rect(0,0,partW,partH));
    cv::Mat part2 = image(cv::Rect(partW,0,partW,partH));
    cv::Mat part3 = image(cv::Rect(0,partH,partW,partH));
    cv::Mat part4 = image(cv::Rect(partW,partH,partW,partH));

    /*creo la nuova immagine e concateno le 4 parti*/
    cv::Mat shufIm;
    cv::Mat temp;
    cv::hconcat(part3,part1,shufIm);
    cv::hconcat(part2,part4,temp);
    cv::vconcat(shufIm,temp,shufIm);

    /*Shuffled channels*/
    std::vector<cv::Mat> ch;
    cv::split(image,ch);  //divide l'immagine nei 3 canali
    cv::Mat shufChIm;
    std::vector<cv::Mat> new_ch;
    new_ch.push_back(ch[2]);  //scambia B con R 
    new_ch.push_back(ch[0]);  //scambia G con B 
    new_ch.push_back(ch[1]);  //scambia R con G 
    cv::merge(new_ch,shufChIm);

    /*Shuffled channels without skip & merge*/
    cv::Mat N(image.rows,image.cols,image.type());

    for(int i=0;i<N.rows;i++){
      for(int j=0;j<N.cols;j++){
        N.data[(i*N.cols+j)*N.elemSize()]=image.data[(i*N.cols+j)*N.elemSize()+2*N.elemSize1()]; //B-->R
        N.data[(i*N.cols+j)*N.elemSize()+1]=image.data[(i*N.cols+j)*N.elemSize()]; //G-->B
        N.data[(i*N.cols+j)*N.elemSize()+2]=image.data[(i*N.cols+j)*N.elemSize()+N.elemSize1()];  //R-->G

      }

    }
    bool bayer = false;
    if(bayer){
      /*Bayer images Demosaicing 1*/
      if(args.image_name.find("BGGR")!=std::string::npos || args.image_name.find("GBRG")!=std::string::npos || args.image_name.find("RGGB")!=std::string::npos){
        cv::Mat gd(image.rows/2, image.cols/2, CV_8UC1);

        for(int r=0; r<gd.rows; r++){
          for(int c=0; c<gd.cols; c++){

            /*coordinate del blocco 2x2*/
            int orig_v = r*2;
            int orig_u = c*2;

            /*leggo i 4 pixels del blocco*/
            int upperleft = image.data[orig_v*image.cols+orig_u];
            int upperright = image.data[orig_v*image.cols+orig_u+1];
            int lowerleft = image.data[(orig_v+1)*image.cols+orig_u];
            int lowerright = image.data[(orig_v+1)*image.cols+orig_u+1];

            /*calcolo il tono di grigio in base al pattern*/
            if(args.image_name.find("BGGR")!=std::string::npos)
              gd.data[(r*gd.cols+c)] = (upperright+lowerleft)/2;
            if(args.image_name.find("GBRG")!=std::string::npos)
              gd.data[(r*gd.cols+c)] = (upperleft+lowerright)/2;
            if(args.image_name.find("RGGB")!=std::string::npos)
              gd.data[(r*gd.cols+c)] = (upperright+lowerleft)/2;
          

          }
        }
        /*Bayer Demosaicing 2*/

        cv::Mat gd1(image.rows,image.cols,CV_8UC1);

        for(int r=0;r<gd1.rows;r++){
          for(int c=0;c<gd1.cols;c++){
            /*coordinate del blocco 2x2*/
            int orig_v= r*2;
            int orig_u = c*2;

	          //leggo i 4 pixel del pattern
	          int upperleft  = image.data[orig_v*image.cols+orig_u];
	          int upperright = image.data[orig_v*image.cols+orig_u + 1];
	          int lowerleft  = image.data[(orig_v+1)*image.cols+orig_u];
	          int lowerright = image.data[(orig_v+1)*image.cols+orig_u+1];

            /*calcolo del tono di grigio*/
            if(args.image_name.find("BGGR")!=std::string::npos)
              gd1.data[r*image.cols+c] = 0.3*float(lowerright)+0.59*float(upperright+lowerleft)/2 + 0.11*(upperleft);
            if(args.image_name.find("GBRG")!=std::string::npos)
              gd1.data[r*image.cols+c] = 0.3*float(lowerleft)+0.59*float(upperleft+lowerright)/2 + 0.11*(upperright);
            if(args.image_name.find("RGGB")!=std::string::npos)
              gd1.data[r*image.cols+c] = 0.3*float(upperleft)+0.59*float(upperright+lowerleft)/2 + 0.11*(lowerright);
          }
        }
        /*Bayer Demosaicing SIMPLE*/

        cv::Mat s(image.rows,image.cols,CV_8UC3);

        for(int r=0;r<s.rows;r++){
          for(int c=0;c<s.cols;c++){
            /*coordinate del blocco 2x2*/
            int orig_v= r;
            int orig_u = c;

            //leggo i 4 pixel del pattern
	          int upperleft  = image.data[orig_v*image.cols+orig_u];
	          int upperright = image.data[orig_v*image.cols+orig_u + 1];
	          int lowerleft  = image.data[(orig_v+1)*image.cols+orig_u];
	          int lowerright = image.data[(orig_v+1)*image.cols+orig_u+1];

            /*Il pattern cambia di continuo.
            ES. RGGB 
            R G R G R G 
	          G B G B G B
	          R G R G R G
	          G B G B G B
	    
	          r,c=0,0 -> RGGB
	          r,c=1,0 -> GBRG
	          r,c=0,1 -> GRBG
	          r,c=1,1 -> BGGR
	    
	          ecc. ecc.*/

            bool isRGGB = args.image_name.find("RGGB")!=std::string::npos;
            bool isGBRG = args.image_name.find("GBRG")!=std::string::npos;
            bool isBGGR = args.image_name.find("BGGR")!=std::string::npos;

            //RGGB
            if((isRGGB && r%2==0 && c%2==0) || (isGBRG && r%2==1 && c%2==1) || (isBGGR && r%2==1 && c%2==1)){
              s.data[(r*s.cols+c)*s.elemSize()] = lowerright; //chB
              s.data[(r*s.cols+c)*s.elemSize()+1] = (upperright+lowerleft)/2; //chG
              s.data[(r*s.cols+c)*s.elemSize()+2] = upperleft;  //chR
            }
            //GBRG
            if((isRGGB && r%2==1 && c%2==0) || (isGBRG && r%2==0 && c%2==0) || (isBGGR && r%2==0 && c%2==1)){
              s.data[(r*s.cols+c)*s.elemSize()] = upperright; //chB
              s.data[(r*s.cols+c)*s.elemSize()+1] = (upperleft+lowerright)/2; //chG
              s.data[(r*s.cols+c)*s.elemSize()+2] = lowerleft;  //chR
            }
            //GRBG
            if((isRGGB && r%2==0 && c%2==1) || (isGBRG && r%2==1 && c%2==1) || (isBGGR && r%2==1 && c%2==0)){
              s.data[(r*s.cols+c)*s.elemSize()] = lowerleft; //chB
              s.data[(r*s.cols+c)*s.elemSize()+1] = (upperleft+lowerright)/2; //chG
              s.data[(r*s.cols+c)*s.elemSize()+2] = upperright;  //chR
            }
            //BGGR
            if((isRGGB && r%2==1 && c%2==1) || (isGBRG && r%2==0 && c%2==1) || (isBGGR && r%2==0 && c%2==0)){
              s.data[(r*s.cols+c)*s.elemSize()] = upperleft; //chB
              s.data[(r*s.cols+c)*s.elemSize()+1] = (upperright+lowerleft)/2; //chG
              s.data[(r*s.cols+c)*s.elemSize()+2] = lowerright;  //chR
            }
          }
        }


        cv::namedWindow("Bayer", cv::WINDOW_NORMAL);
        cv::imshow("Bayer", image);
        cv::namedWindow("Bayer1", cv::WINDOW_NORMAL);
        cv::imshow("Bayer1", gd);
        cv::namedWindow("Bayer2", cv::WINDOW_NORMAL);
        cv::imshow("Bayer2", gd1);
        cv::namedWindow("Bayer SIMPLE", cv::WINDOW_NORMAL);
        cv::imshow("Bayer SIMPLE", s);
      }
    }



    /////////////////////

    //display image
    /*not if bayer image*/
    if(!bayer){
      cv::namedWindow("original image", cv::WINDOW_NORMAL);
      cv::imshow("original image", image);
      cv::namedWindow("newer image", cv::WINDOW_NORMAL);
      cv::imshow("newer image", A);

      //cv::namedWindow("created image", cv::WINDOW_NORMAL);
      //cv::imshow("created image", B);
      cv::namedWindow("Subsampled image", cv::WINDOW_NORMAL);
      cv::imshow("Subsampled image", subsIm);
      cv::namedWindow("Flipped image", cv::WINDOW_NORMAL);
      cv::imshow("Flipped image", flipIm);
      cv::namedWindow("SubH image", cv::WINDOW_NORMAL);
      cv::imshow("SubH image", subsHIm);
      cv::namedWindow("SubW image", cv::WINDOW_NORMAL);
      cv::imshow("SubW image", subsWIm);
      cv::imshow("SubH image", subsHIm);
      cv::namedWindow("Vertically Flipped image", cv::WINDOW_NORMAL);
      cv::imshow("Vertically Flipped image", flipVIm);
      cv::namedWindow("Cropped image", cv::WINDOW_NORMAL);
      cv::imshow("Cropped image",croppedIm);
      cv::namedWindow("Padded image", cv::WINDOW_NORMAL);
      cv::imshow("Padded image", paddedIm);
      cv::namedWindow("Shuffled image", cv::WINDOW_NORMAL);
      cv::imshow("Shuffled image", shufIm);
      cv::namedWindow("Shuffled channels", cv::WINDOW_NORMAL);
      cv::imshow("Shuffled channels", shufChIm);
      cv::namedWindow("Shuffled channels 2", cv::WINDOW_NORMAL);
      cv::imshow("Shuffled channels 2", N);
    }

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

  while ((c = getopt (argc, argv, "hi:t:x:y:w:a:p:")) != -1)
    switch (c)
    {
      case 't':
        args.wait_t = atoi(optarg);
        break;
      case 'i':
        args.image_name = optarg;
        break;
      case 'x':
        args.x=atoi(optarg);
        break;
      case 'y':
        args.y=atoi(optarg);
        break;
      case 'w':
        args.width=atoi(optarg);
        break;
      case 'a':
        args.height=atoi(optarg);
        break;
      case 'p':
        args.padding=atoi(optarg);
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

#endif


