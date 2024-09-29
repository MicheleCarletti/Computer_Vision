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

void gaussianKrnl(float sigma, int r, cv::Mat& krnl);

void GaussianBlur(const cv::Mat& src, float sigma, int r, cv::Mat& out, int stride=1);

void sobel3x3(const cv::Mat& src, cv::Mat& magn, cv::Mat& dir);

template <typename T>
float bilinear(const cv::Mat& src, float r, float c);

void findPeaks(const cv::Mat& magn, const cv::Mat& dir, cv::Mat& out);

int doubleTh(const cv::Mat& magn, cv::Mat& out, float t1, float t2);


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
  float gsigma = 1; //deviazione standard filtro gaussiano
  int gkr = 2; //raggio filtro gaussiano
  float tl = 50.0;  //soglia bassa
  float th = 150.0; //soglia alta

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

    cv::Mat blurred;

    // GAUSSIAN FILTERING
    cv::Mat o, odis;

    GaussianBlur(grey,gsigma,gkr,o);
    cv::convertScaleAbs(o, odis);

    cv::namedWindow("Gaussian Blur", cv::WINDOW_NORMAL);
    cv::imshow("Gaussian Blur", odis);
    

    // SOBEL FILTERING
    // void cv::Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)

    cv::Mat m;
    cv::Mat d(grey.size(),CV_32FC1);
    sobel3x3(grey,m,d);
    cv::namedWindow("Gradient Magnitude", cv::WINDOW_NORMAL);
    cv::imshow("Gradient Magnitude", m);
    /*trick to display orientation*/
    cv::Mat adjMap;
    cv::convertScaleAbs(d, adjMap, 255 / (2*CV_PI));
    cv::Mat falseColorMap;
    cv::applyColorMap(adjMap, falseColorMap, cv::COLORMAP_JET);

    cv::namedWindow("Gradient Direction", cv::WINDOW_NORMAL);
    cv::imshow("Gradient Direction", falseColorMap);

    //NON MAXIMA SUPPRESSION
    cv::Mat thin;
    m.convertTo(m,CV_32FC1);
    d.convertTo(d,CV_32FC1);

    findPeaks(m,d,thin);
    cv::namedWindow("Thinned Image", cv::WINDOW_NORMAL);
    cv::imshow("Thinned Image", thin);

    //CANNY TRESHOLD
    cv::Mat edgeimg;
    thin.convertTo(thin,CV_32FC1);

    int result = doubleTh(thin,edgeimg,tl,th);
    if(result == 1){
      std::cerr<<"La soglia bassa non può superare quella alta!"<<std::endl;
    }
    cv::namedWindow("Edge", cv::WINDOW_NORMAL);
    cv::imshow("Edge", edgeimg);
   



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
      case 'G':
      gsigma+=1;
      std::cout<<"Sigma: "<<gsigma<<std::endl;
      break;
      case 'g':
      if(gsigma>1){
        gsigma-=1;
      }
      std::cout<<"Sigma: "<<gsigma<<std::endl;
      break;
      case 'R':
      gkr+=1;
      std::cout<<"Radius: "<<gkr<<std::endl;
      break;
      case 'r':
      if(gkr!=1){
        gkr-=1;
      }
      std::cout<<"Radius: "<<gkr<<std::endl;
      break;
      case 'd':
	    cv::destroyAllWindows();
	    break;
      case 'p':
	    std::cout << "Mat = "<< std::endl << image << std::endl;
	    break;
      case 'L':
      if(tl<th)
        tl = tl+10;
      std::cout<<"TL: "<<tl<<std::endl;
      break;
      case 'l':
      if(tl>10)
        tl = tl-10;
      std::cout<<"TL: "<<tl<<std::endl;
      break;
      case 'A':
      th = th+10;
      std::cout<<"TH: "<<th<<std::endl;
      break;
      case 'a':
      if(th>10 && th>tl+10)
        th = th-10;
      std::cout<<"TH: "<<th<<std::endl;
      break;
      case 'k':
      {
        static int sindex=0;
        int values[]={3, 5, 7, 11 ,13};
        ksize = values[++sindex%5];
        std::cout << "Setting Kernel size to: " << ksize << std::endl;
      }
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

void gaussianKrnl(float sigma, int r, cv::Mat& krnl){
  /*genera un kernel gaussiano dato sigma e il raggio*/

  int size=2*r+1; //dimensione kernel gaussiano
  krnl.create(size,1, CV_32FC1);  //definisco il kernel come vettore colonna float 32
  float sum=0;

  /*calcolo il valore del peso per ogni elemento del kernel*/
  for(int i=-r; i<=r; i++){
    float weight = exp(-0.5*i*i / (sigma*sigma));
    krnl.at<float>(i+r,0) = weight;
    sum += weight;
  }

  /*normalizzo il kernel*/
  krnl /= sum;
  std::cout<<"Vertical Gaussian Kernel: "<<krnl<<std::endl;
}

void GaussianBlur(const cv::Mat& src, float sigma, int r, cv::Mat& out, int stride){
  /*applica il kernel gaussiano tramite convoluzione*/

  cv::Mat v_gk, h_gk; //dichiaro i kernel verticale e orizzontale
  gaussianKrnl(sigma,r,v_gk);  //calcolo il kernel gaussiano verticale
  h_gk = v_gk.t();  //faccio la trasposta per ottenere quello orizzontale

  cv::Mat res;
  myfilter2D(src,h_gk,res,stride);  //applico il kernel orizzontale e memorizzo temporaneamente in res
  cv::convertScaleAbs(res,res); //converto res nel range 0-255
  myfilter2D(res,v_gk,out,stride);  //applico il kernel verticale
  
}

void sobel3x3(const cv::Mat& src, cv::Mat& magnitude, cv::Mat& direction){
  /*applica un Sobel per calcolare la magnitude e la direction del gradiente*/
  
  cv::Mat gx, gy, agx, agy, ag;
  cv::Mat my_gx, my_gy;
  cv::Mat h_sobel = (cv::Mat_<float>(3,3) << 1, 0, -1,
    2, 0, -2,
    1, 0, -1);
  cv::Mat v_sobel = h_sobel.t();
  myfilter2D(src, h_sobel, my_gx, 1); //calcolo derivata orizzontale
  myfilter2D(src, v_sobel, my_gy, 1); //calcolo derivata verticale
  my_gx.convertTo(gx, CV_32FC1);
  my_gy.convertTo(gy, CV_32FC1);
  // compute magnitude
  cv::pow(gx.mul(gx) + gy.mul(gy), 0.5, magnitude);
  // compute orientation
  //cv::Mat orient(magnitude.size(), CV_32FC1);
  float *dest = (float *)direction.data;
  float *srcx = (float *)gx.data;
  float *srcy = (float *)gy.data;
  float *magn = (float *)magnitude.data;
  for(int i=0; i<gx.rows*gx.cols; ++i)
    dest[i] = magn[i]>50 ? atan2f(srcy[i], srcx[i]) + 2*CV_PI: 0;
  // scale on 0-255 range
  cv::convertScaleAbs(gx, agx);
  cv::convertScaleAbs(gy, agy);
  cv::convertScaleAbs(magnitude, magnitude);
  /*
  cv::namedWindow("sobel verticale", cv::WINDOW_NORMAL);
  cv::imshow("sobel verticale", agx);
  cv::namedWindow("sobel orizzontale", cv::WINDOW_NORMAL);
  cv::imshow("sobel orizzontale", agy);*/

}

template <typename T>
float bilinear(const cv::Mat& src, float r, float c){

  float yDist = r - static_cast<long>(r);
  float xDist = c - static_cast<long>(c);

  float value =
    src.at<T>(r,c)*(1-yDist)*(1-xDist) +
    src.at<T>(r+1,c)*(yDist)*(1-xDist) +
    src.at<T>(r,c+1)*(1-yDist)*(xDist) +
    src.at<T>(r+1,c+1)*yDist*xDist;

  return value;
}

void findPeaks(const cv::Mat& magn, const cv::Mat& dir, cv::Mat& out){
  /*Applica Non-Maxima suppression*/
  int rows = magn.rows;
  int cols = magn.cols;

  out.create(rows, cols, CV_32FC1);
  out.setTo(0);

  /*scorro l'immagine della magnitude e della direction*/
  for(int y=1;y<rows-1;y++){
    for(int x=1;x<cols-1;x++){

      /*valore dell'angolo e del modulo*/
      float angle=dir.at<float>(y,x);
      float m=magn.at<float>(y,x);

      /*calcola la posizione dei pixel adiacenti nella direzione opposta e arrotonda*/
      
      /*int x1=x+static_cast<int>(cos(angle)+0.5);
      int y1=y-static_cast<int>(sin(angle)+0.5);
      int x2=x-static_cast<int>(cos(angle)+0.5);
      int y2=y+static_cast<int>(sin(angle)+0.5);*/
      float x1=x+cos(angle);
      float y1=y+sin(angle);
      float x2=x-cos(angle);
      float y2=y-sin(angle);
      
      /*verifica se il pixel corrente ha un valore maggiore dei sei vicini in direzione opposta*/
      /*if(x1>=0 && x1<cols && y1>=0 && y1<rows && m>magn.at<float>(y1,x1)){
        out.at<float>(y,x)=m;
      }else if(x2>=0 && x2<cols && y2>=0 && y2<rows && m>magn.at<float>(y2,x2)){
        out.at<float>(y,x)=m;
      }*/
      float e1=bilinear<float>(magn,y1,x1);
      float e2=bilinear<float>(magn,y2,x2);

      if(m<e1 || m<e2){
        m=0;
      }     
      out.at<float>(y,x)=m;
    }
  }

  /*converto nel range 0-255*/
  cv::convertScaleAbs(out,out);
}
int doubleTh(const cv::Mat& magn, cv::Mat& out, float t1, float t2){
  /*applica soglia di isterisi*/
  out.create(magn.size(),CV_8UC1);

  if(t1 >= t2){ //se la soglia bassa è maggiore o uguale a quella alta errore!
    return 1;
  }

  int tm = t1 + (t2-t1)/2; //valore medio per pixel con valore compreso tra t1 e t2

  std::vector<cv::Point2i> strong;  //vettore punti forti (>=t2)
  std::vector<cv::Point2i> low; //vettore punti deboli (t1..t2)

  /*scorro l'immagine della magnitude*/
  for(int r=0; r<magn.rows; r++){
    for(int c=0; c<magn.cols; c++){

      if(magn.at<float>(r,c) >= t2){  //se trovo un punto forte
        out.at<uint8_t>(r,c) = 255; //porto il pixel di out a 255
        strong.push_back(cv::Point2i(c,r)); //salvo il punto nel vettore strong
      }
      else if(magn.at<float>(r,c) <= t1){ //se trovo un punto minore di t1
        out.at<uint8_t>(r,c) = 0; //porto il pixel di out a 0
      }
      else{ //valore intermedio (t1..t2)
        out.at<uint8_t>(r,c) = tm;  //assegno al pixel di out un valore medio
        low.push_back(cv::Point2i(c,r));  //salvo il punto nel vettore low
      }
    }
  }

  /*scorro il vettore dei punti forti*/
  while(!strong.empty()){
    cv::Point2i p = strong.back();  //prelevo l'ultimo punto forte
    strong.pop_back();  //lo rimuovo dal vettore
    /*guardo in un intorno del punto forte*/
    for(int dr=-1; dr<=1; dr++){
      for(int dc=-1; dc<=1; dc++){
        int nr = p.y + dr;
        int nc = p.x + dc;
        if(nr >= 0 && nr < out.rows && nc >= 0 && nc < out.cols && out.at<uint8_t>(nr,nc) == tm){ //se trovo un punto debole
          out.at<uint8_t>(nr,nc) = 255; //lo promuovo a 255 (punto forte)
          strong.push_back(cv::Point2i(nc,nr)); //lo aggiungo al vettore dei punti forti
        } 
      }
    }
  }

  /*scorro il vettore dei punti deboli */
  while(!low.empty()){
    cv::Point2i p = low.back(); //prelevo l'ultimo punto debole
    low.pop_back(); //lo rimuovo dal vettore
    if(out.at<uint8_t>(p.y,p.x) < 255){ //se non è stato promosso precedentemente (punto debole isolato)
      out.at<uint8_t>(p.y,p.x) = 0; //porto il pixel di out a 0
    }
  }

  return 0;
}


#endif


