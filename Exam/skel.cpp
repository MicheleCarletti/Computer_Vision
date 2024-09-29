/*STUDENTE: Michele Carletti
MATRICOLA: 347713*/

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>

//options

struct ArgumentList {
  std::string image_name;                   //!< image file name
  std::string points_name;                   //!< points file name
  int dist_t;                               //!< RANSAC distance
  int wait_t;                               //!< waiting time
  double outp;                              //!< outliers percentage
};

bool ParseInputs(ArgumentList& args, int argc, char **argv);


double Distance2Line(cv::Point line_start, cv::Point line_end, cv::Point point)
{
  double normalLength = hypot(line_end.x - line_start.x, line_end.y - line_start.y);
  return fabs((double)((point.x - line_start.x) * (line_end.y - line_start.y) - (point.y - line_start.y) * (line_end.x - line_start.x)) / normalLength);
}

int main(int argc, char **argv)
{
  int frame_number = 0;
  char frame_name[256];
  bool exit_loop = false;
  int imreadflags = cv::IMREAD_COLOR; 

  std::cout<<"Simple program."<<std::endl;

  srand(time(0)); // initialize random number generator

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

    // OUTPUT IMAGE
    cv::Mat out = image/3; 

    /**** YOUR CODE HERE ****/

    // STEP 1: read points
    std::vector<cv::Point2i> points;
    std::ifstream lf(args.points_name);
    if(!lf)
    {
      std::cerr << "Failed to open " << args.points_name << std::endl;
    }
    
    while(lf)
    {
        cv::Point2i p;
        lf >> p.y >> p.x;
        if(lf)
            points.push_back(p);
    }


    // STEP 2 & 3: RANSAC
    double op = args.outp;
    std::cout<< "Prob: "<< op<< std::endl;
    float v = 1.0-op; //prob inlier
    int N = int(log(1-0.99)/log(1 -(v*v))); //iterazioni RANSAC
    std::cout<< "N: "<< N<< std::endl;

    std::vector<cv::Point2i> inliers;
    std::vector<cv::Point2i> outliers;
    std::vector<cv::Point2i> best_inliers;
    std::vector<cv::Point2i> best_outliers;
    int best_inliers_count = 0;

    /*ciclo RANSAC*/
    for(int i=0; i<N; i++){

      inliers.clear();
      outliers.clear();

      cv::Point2i p1, p2;
      p1 = points[rand()%points.size()];
      p2 = points[rand()%points.size()];

      int count_inliers = 0;

      /*per ogni punto calcolo la distanza tra esso e la retta passante per p1 e p2*/
      for(int j=0; j<points.size(); j++){
        double dist = Distance2Line(p1,p2,points[j]);
        if(dist <= args.dist_t){  //se la distanza è inferiore al dist_t il punto è inlier
          inliers.push_back(points[j]); 
          count_inliers++;
        }else{  //altrimenti è outlier
          outliers.push_back(points[j]);
        }
      }

      /*se l'iterazione corrente produce un numero maggiore di inliers rispetto alla precedente
      aggiorno il conteggio dei best inliers e i vettori di punti*/
      if(best_inliers_count < count_inliers){
        best_inliers_count = count_inliers;
        best_inliers = inliers;
        best_outliers = outliers;
      }
      /*definisco una soglia oltre la quale posso terminare prima RANSAC*/
      double t = v*points.size();
      if(best_inliers.size() >= t){
        std::cout<< "Real number of iterations: "<< i+1<<std::endl;
        break;
      }
   
    }

    /*prelevo due punti casuali per generare la best line*/
    cv::Point2i rp1, rp2;
    rp1 = best_inliers[rand()%best_inliers.size()];
    rp2 = best_inliers[rand()%best_inliers.size()];

    /*mostro gli inliers i rosso*/
    for(int i=0; i<best_inliers.size(); i++){
      cv::circle(out, best_inliers[i], 3, cv::Scalar(0,0,255), -1);
      
    }
    /*mostro gli outliers in blu*/
    for(int i=0; i<best_outliers.size(); i++){
      cv::circle(out, best_outliers[i], 3, cv::Scalar(255,0, 0), -1);
    }

    /*calcolo m e q della best line e genero due nuovi punti con coord x fuori dall'immagine*/
    double m = (rp2.y - rp1.y)/(rp2.x - rp1.x);
    double q = -m*rp1.x + rp1.y;
    std::cout<< "m: "<< m<< std::endl;
    std::cout<< "q: "<< q<< std::endl;

    int newy1 = m*rp1.x + q;
    int newy2 = m*rp2.x + q; 

    cv::Point2i np1, np2;
    np1.x = -10000;
    np1.y = newy1;
    np2.x = +10000;
    np2.y = newy2;

    cv::line(out,np1,np2, cv::Scalar(0,255,0), 2, cv::LINE_AA);


    cv::namedWindow("output image", cv::WINDOW_NORMAL); // cv::WINDOW_AUTOSIZE if you want the window to adapt to image size
    cv::imshow("output image", out);

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
      case 'q':
	exit_loop = 1;
	break;
    }
    frame_number++;
  }

  return 0;
}


#include <unistd.h>
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  int c;
  args.dist_t = 30;
  args.wait_t = 0;
  args.outp = 0.33;

  while ((c = getopt (argc, argv, "hi:t:d:p:o:")) != -1)
    switch (c)
    {
      case 'd':
        args.dist_t = atoi(optarg);
        break;
      case 't':
        args.wait_t = atoi(optarg);
        break;
      case 'p':
        args.points_name = optarg;
        break;
      case 'o':
        args.outp = atof(optarg);
        break;
      case 'i':
        args.image_name = optarg;
        break;
      case 'h':
      default:
        std::cout<<"Allowed options:"<<std::endl<<
          "   -h                       produce help message"<<std::endl<<
          "   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
          "   -p arg                   points file name. "<<std::endl<<
          "   -d arg                   RANSAC distance (default 30)"<<std::endl<<
          "   -o arg                   outliers percentage (default 0.33)"<<std::endl<<
          "   -t arg                   wait before next frame (ms)"<<std::endl<<std::endl;
        return false;
    }
  return true;
}



