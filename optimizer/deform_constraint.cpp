//
// Created by pointer on 18-3-8.
//

#include "deform_constraint.h"

DeformConstraint::DeformConstraint(
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
    double img_obs, int k_vec_num,
    Eigen::Matrix<double, 3, 1> vec_M, Eigen::Matrix<double, 3, 1> vec_D,
    double epi_A, double epi_B, double fx, double fy, double dx, double dy,
    Eigen::Matrix<double, 3, 1> light_vec,
    Eigen::Matrix<double, Eigen::Dynamic, 1> weight) : pattern_(pattern) {
  this->img_obs_ = img_obs;
  this->k_vec_num_ = k_vec_num;
  this->vec_M_ = vec_M;
  this->vec_D_ = vec_D;
  this->epi_A_ = epi_A;
  this->epi_B_ = epi_B;
  this->f_x_ = fx;
  this->f_y_ = fy;
  this->d_x_ = dx;
  this->d_y_ = dy;
  this->light_vec_ = light_vec;
  this->weight_ = weight;
}

DeformConstraint::DeformCostFunction* DeformConstraint::Create(
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
    double img_obs, int k_vec_num,
    Eigen::Matrix<double, 3, 1> vec_M, Eigen::Matrix<double, 3, 1> vec_D,
    double epi_A, double epi_B, double fx, double fy, double dx, double dy,
    Eigen::Matrix<double, 3, 1> light_vec,
    Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr,
    double * vertex_set,
    std::vector<double*>* parameter_blocks
) {

  // Calculate weight
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight
      = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_vec_num, 1);
  double sum_val = 0;
  for (int i = 0; i < k_vec_num; i++) {
    weight(i) = pow((1 - vertex_nbr(i, 1) / vertex_nbr(k_vec_num, 1)), 2);
    sum_val += weight(i);
  }
  weight = weight / sum_val;

  DeformConstraint* constraint = new DeformConstraint(
      pattern, img_obs, k_vec_num, vec_M, vec_D, epi_A, epi_B, fx, fy, dx, dy,
      light_vec, weight);
  DeformCostFunction* cost_function = new DeformCostFunction(constraint);
  // Add parameter blocks
  parameter_blocks->clear();
  for (int i = 0; i < k_vec_num; i++) {
    int idx = vertex_nbr(i, 0);
    parameter_blocks->push_back(&(vertex_set[idx*4]));
    cost_function->AddParameterBlock(4);
  }
  cost_function->SetNumResiduals(1);
  return (cost_function);
}