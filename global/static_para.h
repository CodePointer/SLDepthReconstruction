//
// Created by pointer on 17-12-6.
//

#ifndef DEPTHOPTIMIZATION_STATIC_PARA_H
#define DEPTHOPTIMIZATION_STATIC_PARA_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

const int kCamHeight = 1024;
const int kCamWidth = 1280;
const int kCamVecSize = kCamHeight * kCamWidth;
const int kProHeight = 800;
const int kProWidth = 1280;

const double kDepthMin = 20.0;
const double kDepthMax = 60.0;
const int kGridSize = 15;
const int kFrameNum = 90;
const uchar kMaskIntensityThred = 10;
const int kMaskMinAreaThred = 20;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ImgMatrix;

namespace my {
  const uchar VERIFIED_FALSE = 0;
  const uchar VERIFIED_TRUE = 1;
  const uchar INITIAL_FALSE = 2;
  const uchar INITIAL_TRUE = 3;
  const uchar MARKED = 4;
  const uchar NEIGHBOR_4 = 5;
  const uchar NEIGHBOR_8 = 6;

  const uchar DIREC_UP_LEFT = 0;
  const uchar DIREC_UP = 1;
  const uchar DIREC_UP_RIGHT = 2;
  const uchar DIREC_RIGHT = 3;
  const uchar DIREC_DOWN_RIGHT = 4;
  const uchar DIREC_DOWN = 5;
  const uchar DIREC_DOWN_LEFT = 6;
  const uchar DIREC_LEFT = 7;
}

struct CamMatSet {
  cv::Mat img_obs;
  cv::Mat img_est;
  cv::Mat x_pro;
  cv::Mat y_pro;
  cv::Mat depth;
  cv::Mat mask;
};

struct CalibSet {
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> cam;
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> pro;
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R;
  Eigen::Matrix<double, 3, 1> t;
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> cam_mat;
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> pro_mat;
  Eigen::Matrix<double, 3, Eigen::Dynamic> M;
  Eigen::Matrix<double, 3, 1> D;
  Eigen::Matrix<double, 3, 1> light_vec_;
};

#endif //DEPTHOPTIMIZATION_STATIC_PARA_H
