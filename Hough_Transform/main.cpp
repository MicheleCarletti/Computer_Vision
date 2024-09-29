//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

struct ArgumentList {
  std::string image_name;		  //!< image file name
  int wait_t;                    //!< waiting time
  bool verbose;                 //!< show additional info
};

bool ParseInputs(ArgumentList& args, int argc, char **argv);

void Hough(const cv::Mat& src, cv::Mat& out, bool debug);

int main(int  argc, char** argv){
    
    int frame_number = 0;
    char frame_name[256];
    bool exit_loop = false;
    int th = 2.5; //threshold for predicted line 1.0, 1.5, 2.0, 2.5

    ArgumentList args;
    if(!ParseInputs(args, argc, argv)){
        return 1;
    }

    while(!exit_loop){
        /*generating file name*/
        if(args.image_name.find('%') != std::string::npos)  //multi frame case
            sprintf(frame_name, (const char*) (args.image_name.c_str()), frame_number);
        else    //single frame case
            sprintf(frame_name, "%s", args.image_name.c_str());
        
        /*opening file*/
        cv::Mat image = cv::imread(frame_name);
        if(image.empty()){
            std::cerr<<"Unable to open "<<frame_name<<std::endl;
            return 1;
        }

        /*display image*/
        cv::namedWindow("Original image", cv::WINDOW_NORMAL);
        cv::imshow("Original image", image);

        //PROCESSING
        cv::Mat grey(image.size(), CV_8UC1);
        cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
        if(args.verbose){
            cv::namedWindow("Greyscale", cv::WINDOW_NORMAL);
            cv::imshow("Greyscale", grey);
        }

        cv::Mat hough;
        Hough(grey, hough, args.verbose);
        cv::Mat hough_draw;
        hough.convertTo(hough_draw, CV_8UC1, 1./16);
        cv::namedWindow("Hough Transform", cv::WINDOW_NORMAL);
        cv::imshow("Hough Transform", hough_draw);

        /*find the global maximum in Hough image*/
        cv::Point max_coord;

        /*cancello i pixel in un intorno di +- 30px del punto massimo*/
        
        for(int i=0; i<3; i++){ 
            cv::minMaxLoc(hough,NULL,NULL,NULL,&max_coord);
            std::cout<<"Max point: r = "<<max_coord.y<<" c = "<<max_coord.x<<std::endl;
            for(int r=0; r<hough.rows; r++){
                for(int c=max_coord.x-30; c<max_coord.x+30; c++){
                    hough.at<int>(r,c) = 0;
                }
            }

        }


        cv::Mat result(grey.size(),CV_8UC1);

        /*compute theta and rho*/
        double theta = max_coord.y*(M_PI/180); 
        double rho = max_coord.x;   

        /*draw the main line*/
        cv::Point pt1, pt2;
        double x0 = rho * cos(theta);
        double y0 = rho * sin(theta);

        pt1.x = cvRound(x0 + 1000 * (-sin(theta)));
        pt1.y = cvRound(y0 + 1000 * (cos(theta)));
        pt2.x = cvRound(x0 - 1000 * (-sin(theta)));
        pt2.y = cvRound(y0 - 1000 * (cos(theta)));

        cv::Mat mainl(grey.size(), CV_8UC3);
        mainl = image.clone();
        cv::line(mainl, pt1, pt2, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::namedWindow("Main line", cv::WINDOW_NORMAL);
        cv::imshow("Main line", mainl);

        /*Scan the greyscale image*/
        for(int r=0; r<grey.rows; r++){
            for(int c=0; c<grey.cols; c++){
                double predicted_line = (rho-sin(theta)*r)/(cos(theta));    //compute the line
                /*show the part of image aligned with the line*/
                if(predicted_line + th < grey.at<uchar>(r,c)){ 
                    result.at<uchar>(r,c) = 0;
                }else{
                    result.at<uchar>(r,c) = 255;
                }
            }
        }
        cv::namedWindow("Result", cv::WINDOW_NORMAL);
        cv::imshow("Result", result);

        /*wait for key or timeout*/
        unsigned char key = cv::waitKey(args.wait_t);
        std::cout<<"Key "<<int(key)<<std::endl;

        if(key == 'q'){
            exit_loop = true;
        }

        frame_number++;
        
    }

    return 0;
}

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  int c;
  args.wait_t = 0;
  args.verbose = false;

  while ((c = getopt (argc, argv, "hi:t:v")) != -1)
    switch (c)
    {
    case 't':
	    args.wait_t = atoi(optarg);
	break;
    case 'i':
	    args.image_name = optarg;
	break;
    case 'v':
        args.verbose = true;
    break;
    case 'h':
    default:
	std::cout<<"usage: " << argv[0] << " -i <image_name>"<<std::endl;
	std::cout<<"exit:  type q"<<std::endl<<std::endl;
	std::cout<<"Allowed options:"<<std::endl<<
	  "   -h                       produce help message"<<std::endl<<
	  "   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
	  "   -t arg                   wait before next frame (ms)"<<std::endl<<
      "   -v                       verbose (show additional info)"<<std::endl<<std::endl;
	return false;
    }
  return true;
}

void Hough(const cv::Mat& src, cv::Mat& out, bool debug){
    /*Compute Hough Transform*/

    cv::Mat hist(src.rows, 128, CV_16U, cv::Scalar(0));
    cv::Mat h(360, 400, CV_32S, cv::Scalar(0));

    /*compute src histogram*/
    for(int r=0; r<src.rows; r++){
        for(int c=0; c<src.cols; c++){
            if(c < hist.cols){
                hist.at<u_short>(r, src.at<uchar>(r,c)) += 1;
            }
        }
    }

    if(debug){
        cv::Mat histDraw(hist.size(), CV_8UC1);
        hist.convertTo(histDraw, CV_8UC1);
        cv::namedWindow("Histogram", cv::WINDOW_NORMAL);
        cv::imshow("Histogram", histDraw);
    }

    /*compute Hough from histogram*/
    for(int r =0; r<hist.rows; r++){
        for(int c=0; c<hist.cols; c++){
            for(int rh=0; rh<h.rows; rh++){
                double theta = rh*(M_PI/180);
                int rho = round(cos(theta)*c+sin(theta)*r); 
                if(rho >= 0 && rho <= 400){
                    h.at<int>(rh,rho) += hist.at<uchar>(r,c);
                }
            }
        }
    }

    h.copyTo(out);

}
