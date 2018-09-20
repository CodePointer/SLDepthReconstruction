//
// Created by pointer on 17-12-6.
//

#ifndef DEPTHOPTIMIZATION_STATIC_PARA_H
#define DEPTHOPTIMIZATION_STATIC_PARA_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <glog/logging.h>

const int kCamHeight = 1024;
const int kCamWidth = 1280;
const int kCamVecSize = kCamHeight * kCamWidth;
const int kProHeight = 800;
const int kProWidth = 1280;

const int kIntensityClassNum = 4;
const int kTemporalWindowSize = 4;
const int kNodeBlockSize = 16;
const double kDepthMin = 15.0;
const double kDepthMax = 47.0;
const int kGridSize = 15;
const uchar kMaskIntensityThred = 10;
const int kMaskMinAreaThred = 40;
const double kDepthRad = 2;
const int kNearestPoints = 8; // Used for neighbor interpolation
const int kRegularNbr = 8;
const double kStripDis = 12;
const double kClassNum = 6;

// For k_means
const int kKMBlockHeightNum = 8;
const int kKMBlockWidthNum = 8;
const int kKMBlockHeight = 128;
const int kKMBlockWidth = 160;

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
  cv::Mat img_class;
//  cv::Mat img_class_p;
//  cv::Mat shade_mat;
//  cv::Mat img_est;
  cv::Mat x_pro;
  cv::Mat y_pro;
  cv::Mat depth;
  cv::Mat x_pro_range;
  ImgMatrix pointer;
  cv::Mat mask;
  cv::Mat mesh_mat;
  Eigen::Matrix<double, 2, Eigen::Dynamic> uv_weight;
//  Eigen::Matrix<double, kIntensityClassNum, Eigen::Dynamic> km_center;
  Eigen::Matrix<double, 3, Eigen::Dynamic> norm_vec;
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
