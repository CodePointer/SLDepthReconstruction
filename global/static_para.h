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

const int kGridSize = 15;
const int kFrameNum = 70;
const uchar kMaskIntensityThred = 18;
const int kMastMinAreaThred = 20;

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
  cv::Mat x_pro;
  cv::Mat y_pro;
  cv::Mat depth;
  cv::Mat mask;
};

struct CalibSet {
  Eigen::Matrix<double, 3, 3> cam;
  Eigen::Matrix<double, 3, 3> pro;
  Eigen::Matrix<double, 3, 3> R;
  Eigen::Matrix<double, 3, 1> t;
  Eigen::Matrix<double, 3, 4> cam_mat;
  Eigen::Matrix<double, 3, 4> pro_mat;
  Eigen::Matrix<double, 3, Eigen::Dynamic> M;
  Eigen::Matrix<double, 3, 1> D;
};

#endif //DEPTHOPTIMIZATION_STATIC_PARA_H
