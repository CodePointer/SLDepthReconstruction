//
// Created by pointer on 17-12-12.
//

#include "deform_cost_functor.h"

DeformCostFunctor::DeformCostFunctor(
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> &pattern,
    double img_obs, double img_obs_last, double img_est_last,
    Eigen::Matrix<double, 3, 1> vec_M, Eigen::Matrix<double, 3, 1> vec_D,
    double epi_A, double epi_B, Eigen::Matrix<double, 2, 2> range,
    Eigen::Matrix<double, 2, 1> pos_k,
    double f_x, double f_y, double d_x, double d_y,
    Eigen::Matrix<double, 3, 1> light_vec) : pattern_(pattern) {
  this->img_obs_ = img_obs;
  this->img_obs_last_ = img_obs_last;
  this->img_est_last_ = img_est_last;
  this->vec_M_ = vec_M;
  this->vec_D_ = vec_D;
  this->epi_A_ = epi_A;
  this->epi_B_ = epi_B;
  this->range_ = range;
  this->pos_k_ = pos_k;
  this->f_x_ = f_x;
  this->f_y_ = f_y;
  this->d_x_ = d_x;
  this->d_y_ = d_y;
  this->light_vec_ = light_vec;
}
