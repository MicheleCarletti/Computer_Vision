//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <eigen3/Eigen/Dense>
#include <unistd.h>

//std:
#include <fstream>
#include <iostream>
#include <string>

struct ArgumentList {
    std::string image_name;		    //!< image file name
    int wait_t;                     //!< waiting time
    bool verbose;                   //!< show additional info
};

struct Keypoint {   //contenitore per i keypoints
    int x;
    int y;
    double response;
};

bool ParseInputs(ArgumentList& args, int argc, char **argv);

void HarrisCornerDetection(const cv::Mat& src, cv::Mat& out, double alpha=0.04, int window_size=3, float sigma=2, int max_num = 100);

std::vector<Keypoint> NonMaximaSuppression(const cv::Mat& src,int neighborhood_size=3);

ArgumentList args;

int main(int argc, char** argv){

    int frame_number = 0;
    char frame_name[256];
    bool exit_loop = false;

    

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
        
        
        cv::Mat outImg;

        HarrisCornerDetection(image, outImg);
        cv::namedWindow("Harris Keypoints", cv::WINDOW_NORMAL);
        cv::imshow("Harris Keypoints", outImg);


        //wait for key or timeout
        unsigned char key = cv::waitKey(args.wait_t);
        std::cout<<"key "<<int(key)<<std::endl;
        
        frame_number++;
        /*pensato per lavorare sulla sequenza kitti*/
        if(frame_number == 1101){
            exit_loop = true;
        }


        switch(key)
        {
            case 'q':
                exit(0);
            break;
        }
        frame_number++;
    }
    
    return 1;
}

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
    int c;
    args.wait_t = 0;
    args.verbose = false;

    while ((c = getopt (argc, argv, "hi:t:v")) != -1){
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
            "   -v                       verbose yes (shows additional info)"<<std::endl<<std::endl;
	        return false;
        }

    }

    return true;
}
std::vector<Keypoint> NonMaximaSuppression(const cv::Mat& src,int neighborhood_size){
    /*Applica NMS sull'immagine src, restituisce i keypoints,
      cercando il massimo in un intorno di dimensione neighborhood_size*/
    int hns = neighborhood_size/2;
    std::vector<Keypoint> k;

    for(int r=hns; r<src.rows-hns; r++){    
        for(int c=hns; c<src.cols-hns; c++){
            double curResp = src.at<double>(r,c);
            
            /*verifico se il punto corrente è un massimo locale*/
            bool is_max = true;
            for(int i=-hns; i<=hns; i++){
                for(int j=-hns; j<=hns; j++){
                    /*se il pixel di partenza non è un massimo locale interrompo il ciclo*/
                    if(src.at<double>(r+i,c+j)>curResp){    
                        is_max = false;
                        break;
                    }
                }
                if(!is_max){
                    break;
                }
            }
            /*se è un punto di massimo locale lo aggiungo al vettore keypoints*/
            if(is_max){
                
                Keypoint key;
                key.x = r;
                key.y = c;
                key.response = curResp;
                k.push_back(key);
            }
        }
    }
    return k;
}
void HarrisCornerDetection(const cv::Mat& src, cv::Mat& out, double alpha, int window_size, float sigma, int max_num){
    
    // convert to grey scale for following processings
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);


    /*Calcolo i gradienti usando Sobel*/
    cv::Mat Ix, Iy;
    cv::Sobel(grey, Ix, CV_64F, 1, 0, 3); //gradiente verticale win. 3x3
    cv::Sobel(grey, Iy, CV_64F, 0, 1, 3); //gradiente orizzontale win. 3x3
    
    if(args.verbose){
        cv::namedWindow("HorizG", cv::WINDOW_NORMAL);
        cv::imshow("HorizG", Iy);
        cv::namedWindow("VertG", cv::WINDOW_NORMAL);
        cv::imshow("VertG", Ix);
    }   


    /*Calcolo le componenti della matrice H*/
    cv::Mat Ixx = Ix.mul(Ix);
    cv::Mat Iyy = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    /*Applico un kernel Gaussiano alla somma cumulativa di H*/
    cv::Mat Sxx, Syy, Sxy;
    cv::GaussianBlur(Ixx, Sxx, cv::Size(window_size, window_size),sigma);
    cv::GaussianBlur(Iyy, Syy, cv::Size(window_size, window_size),sigma);
    cv::GaussianBlur(Ixy, Sxy, cv::Size(window_size, window_size),sigma);

    /*Calcolo il determinante e la traccia di H*/
    cv::Mat det = (Sxx.mul(Syy) - Sxy.mul(Sxy));
    cv::Mat trace = (Sxx + Syy);

    /*Calcolo tehta*/
    cv::Mat theta(grey.size(),CV_64F);
    theta = det - alpha*(trace.mul(trace));

    /*trova i keypoints:
        se la risposta di Harris è minore di 0 "rimuovo" il pixel
        applico NMS ai pixel rimasti*/
        
    cv::Mat response = theta.clone();
    for(int r=0; r<theta.rows; r++){
        for(int c=0; c<theta.cols; c++){
            if(theta.at<double>(r,c) < 0){
                response.at<double>(r,c) = 0;
            }
        }
    }

    if(args.verbose){
        cv::Mat show_response;
        response.convertTo(show_response, CV_8UC1);
        cv::namedWindow("Response", cv::WINDOW_NORMAL);
        cv::imshow("Response", show_response);
    }

    std::vector<Keypoint> keypoints;
    keypoints = NonMaximaSuppression(response);

    /*riordina i keypoints in base alla risposta di Harris*/
    std::sort(keypoints.begin(), keypoints.end(), [](const Keypoint& a, const Keypoint& b) {
        return a.response > b.response;
    });
    /*tengo solo i primi max_num*/
    if(keypoints.size() > max_num){
        keypoints.resize(max_num);
    }

    out = src.clone();
    std::cout<<keypoints.size()<<std::endl;
    for(size_t i=0; i<keypoints.size(); i++){
        out.at<cv::Vec3b>(keypoints[i].x,keypoints[i].y)[0] = 0;    //B
        out.at<cv::Vec3b>(keypoints[i].x,keypoints[i].y)[1] = 255;  //G
        out.at<cv::Vec3b>(keypoints[i].x,keypoints[i].y)[2] = 0;    //R
        /*disegno un cerchio attorno al keypoint di raggio 5 colore verde (vuoto: manca il -1 alla fine)*/
        cv::circle(out, cv::Point2d(keypoints[i].y,keypoints[i].x),5,cv::Scalar(0,255,0)); 

    }

}