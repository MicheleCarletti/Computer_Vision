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
  std::string image_name;		    //!< image file name
  int wait_t;                     //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char **argv);

int main(int argc, char** argv){

    int frame_number = 0;
    char frame_name[256];
    bool exit_loop = false;

    ArgumentList args;

    if(!ParseInputs(args, argc, argv)) {
        exit(0);
    }

    while(!exit_loop){

        /*generating frame name*/
        if(args.image_name.find('%') != std::string::npos)  //multi frame case
            sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
        else    //single frame case
            sprintf(frame_name,"%s",args.image_name.c_str());
        
        //opening file
        std::cout<<"Opening "<<frame_name<<std::endl;
        cv::Mat image = cv::imread(frame_name);
        if(image.empty()){
            std::cerr<<"Unable to open "<<frame_name<<std::endl;
            return 1;
        }

        //display image
        cv::namedWindow("Original image", cv::WINDOW_NORMAL);
        cv::imshow("Original image", image);

        //PROCESSING

        std::vector<int> hist(256,0); //array of 256 elements init. 0
        cv::Mat grey(image.size(), CV_8UC1);
        cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
        cv::namedWindow("Greyscale image", cv::WINDOW_NORMAL);
        cv::imshow("Greyscale image", grey);
    
        for(int r=0; r<grey.rows; r++){
            for(int c=0; c<grey.cols; c++){
                hist[grey.at<uchar>(r,c)] += 1;
            }
        }
        std::cout<<"Histogram size:  "<<hist.size()<<std::endl;

        int max=0, id=0;
        for(size_t i=0; i<hist.size(); i++){
            if(hist[i] > max){
                max = hist[i];
                id = i;
            }
                
        }
        std::cout<<"Max value "<<max<<" pixel value "<<id<<std::endl;
        
        cv::Mat h(max+3, hist.size(), CV_8UC1, cv::Scalar(255));

        for(int c=0; c<h.cols; c++){
            bool stop = false;
            int r = h.rows;
            while(!stop){
                if(hist[c] != 0){
                    h.at<uchar>(r,c) = 0;
                    hist[c] -= 1;
                    r--;
                }else{
                    stop = true;
                }
            }
        }

        //display histogram
        cv::namedWindow("Histogram", cv::WINDOW_NORMAL);
        cv::imshow("Histogram", h);

        //wait fof key or timeout
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

    while ((c = getopt (argc, argv, "hi:t:v")) != -1){
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
	        "   -t arg                   wait before next frame (ms)"<<std::endl<<
            "   -v                       verbose yes (shows additional info)"<<std::endl<<std::endl;
	        return false;
        }

    }

    return true;
}