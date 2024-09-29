// std
#include <iostream>
#include <fstream>

// opencv
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>


// eigen
#include <eigen3/Eigen/Core>

// utils
#include "utils.h"

using namespace cv;

void Project(const std::vector<cv::Point3f>& points, const CameraParams& params, std::vector<cv::Point2f>& uv_points);

std::string im_win_name = "Image";
std::string im_win_name_loop = "Image_loop";
std::string im_win_name_spiral = "Image_spiral";

int main(int argc, char **argv) {

  if (argc < 3) 
  {
    std::cerr << "Usage lab5_1 <points_filename> <camera_params_filename>" << std::endl;
    return 0;
  }

  // load point cloud from file
  std::vector<Point3f> points;
  LoadPoints(argv[1], points);

  // load camera params from file
  CameraParams params;
  LoadCameraParams(argv[2], params);

  // 3d visualization
#ifdef USE_OPENCVVIZ
  cv::Mat cloud;
  PointsToMat(points, cloud);

  cv::viz::Viz3d win = Viz3D(params);

  win.showWidget("cloud", cv::viz::WCloud(cloud));

  std::cout << "Press q to exit" << std::endl;
  win.spin();
#endif

  // project 3d points on image
  std::vector<Point2f> uv_points;
  
  Project(points, params, uv_points);

  

  // draw image
  Mat image;
  image = Mat::zeros(params.h, params.w, CV_32FC1);
  DrawPixels(uv_points, image);

  namedWindow(im_win_name, WINDOW_AUTOSIZE );
  imshow(im_win_name, image);
  waitKey(0);
  

  // rotazione intorno all'edificio
  //
  // Provare ad implementare un loop di 8 posizioni sul piano XZ equidistanti dal baricentro dell'edificio, raggio 30m.
  // Per mantenere l'edificio al centro della visuale dobbiamo ruotare l'orientazione della camera di 45 gradi (2*M_PI/8) ad ogni step
  //
  //
  // centro del palazzo sul piano XZ, Y costante

  //baricentro dell'edificio ad altezza fissata
  float bx=0.0, by=-5.0, bz=0.0;

  for(unsigned int i=0;i<points.size();++i)
  {
    bx+=points[i].x;
    bz+=points[i].z;
  }

  bx/=points.size();
  bz/=points.size();

  std::cout<<"Building center "<<bx<<" "<<by<<" "<<bz<<std::endl;


  // L'idea e' di muoversi lungo una circonferenza di raggio radius e centrato nel baricentro dell'edificio
  //
  // Supponiamo di volerci spostare su 16 posizioni equidistanti, possiamo utilizzare una variable angle che da da
  // 0 a 2PI per step costanti (2*M_PI/16), e quindi calcolare la deltaX e deltaZ con seno e coseno.
  //
  float radius = 30.0;
  float angle=0.0;
  int steps = 16;
  namedWindow(im_win_name_loop, WINDOW_AUTOSIZE );
  int i = 0;
  while(1)
  {
    /**
     * YOUR CODE HERE:
     *
     * Calcolare i params opportuni per spostare il punto di vista lungo la circonferenza
     * mantenendo l'orientazione che punti verso l'edificio
     *
     * Utilizzare la funzione PoseToAffine fornita per calcolare i nuovi params
     * void PoseToAffine(float rx, float ry, float rz, float tx, float ty, float tz, cv::Affine3f& affine)
     */
    
    float rx, ry, rz, tx, ty, tz;

    rx= 0;  //nessuna rotazione attorno a x
    ry= angle;  //rotazione attorno a y per mantenere la camera puntata verso l'edificio
    rz= 0;  //nessuna rotazione attorno a z
    tx= bx-radius*sin(angle); //traslazione rispetto a x
    ty= by; //mantengo la y del baricentro
    tz= bz-radius*cos(angle); //traslazione rispetto a z
    PoseToAffine(rx,ry,rz,tx,ty,tz,params.RT);  
    // project 3d points on image
    uv_points.clear();
    Project(points, params, uv_points);

    // draw image
    Mat image_loop;
    image_loop = Mat::zeros(params.h, params.w, CV_32FC1);
    DrawPixels(uv_points, image_loop);

    imshow(im_win_name_loop, image_loop);
    waitKey(200);

    angle+=2*M_PI/float(steps);

    ++i;
    
    if(i == 17) //esco completato un giro 
      break;
  }

  return 0;
}


void Project(const std::vector< Point3f >& points, const CameraParams& params, std::vector< Point2f >& uv_points)
{
  
  Eigen::Matrix<float, 4, 4> RT;
  Affine3f RT_inv = params.RT.inv();
  RT << RT_inv.matrix(0,0), RT_inv.matrix(0,1), RT_inv.matrix(0,2), RT_inv.matrix(0,3), 
     RT_inv.matrix(1,0), RT_inv.matrix(1,1), RT_inv.matrix(1,2), RT_inv.matrix(1,3), 
     RT_inv.matrix(2,0), RT_inv.matrix(2,1), RT_inv.matrix(2,2), RT_inv.matrix(2,3),
     0,                  0,                  0,                  1;

  Eigen::Matrix<float, 3, 4> K;
  K << params.ku,         0, params.u0, 0,
    0, params.kv, params.v0, 0,
    0,         0,         1, 0;

  /**
   * YOUR CODE HERE: project points from 3D to 2D
   * hint: p' = K*RT*P'
   */
  Eigen::Matrix<float, 3, 4> M;
  M = K*RT;
  Eigen::Matrix<float, 3, 1> p; //p 2D image plane in homogenous coords.
  Eigen::Matrix<float, 4, 1> P; //P 3D world in homogenous coords.
  Point2f res;
  
  
  for(int i=0; i<points.size(); i++){
    //std::cout<<"Step: "<<i<<std::endl;   
    
    /*For each point in the world compute the corresponding point
    in the image plane*/

    P << points[i].x, points[i].y, points[i].z, 1.0f;
    p = M*P;

    /*From homogenous coords. to Euclidean ones*/
    res.x  = p[0]/p[2];
    res.y  = p[1]/p[2];

    //std::cout<<"x: "<<res.x<<std::endl;
    //std::cout<<"y: "<<res.y<<std::endl;
    uv_points.push_back(res);
    
    
  }
  //std::cout<<"Points in image plane: "<<uv_points.size()<<std::endl;
  
 

  
}


