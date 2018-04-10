//
// Created by pointer on 18-3-8.
//

#include "deform_constraint.h"

DeformConstraint::DeformConstraint(
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
    double img_obs, Eigen::Matrix<double, 3, 1> vec_M,
    Eigen::Matrix<double, 3, 1> vec_D,
    double epi_A, double epi_B, double fx, double fy, double dx, double dy,
    Eigen::Matrix<double, 3, 1> light_vec,
    int k_vec_num, int k_vec_opt,
    Eigen::Matrix<double, Eigen::Dynamic, 1> weight,
    Eigen::Matrix<double, 3, 1> norm_add, double depth_add) : pattern_(pattern) {
  this->img_obs_ = img_obs;
  this->vec_M_ = vec_M;
  this->vec_D_ = vec_D;
  this->epi_A_ = epi_A;
  this->epi_B_ = epi_B;
  this->f_x_ = fx;
  this->f_y_ = fy;
  this->d_x_ = dx;
  this->d_y_ = dy;
  this->light_vec_ = light_vec;
  this->k_vec_num_ = k_vec_num;
  this->k_vec_opt_ = k_vec_opt;
  this->weight_ = weight;
  this->norm_add_ = norm_add;
  this->depth_add_ = depth_add;
}

DeformConstraint::DeformCostFunction* DeformConstraint::Create(
    ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
    double img_obs, Eigen::Matrix<double, 3, 1> vec_M,
    Eigen::Matrix<double, 3, 1> vec_D,
    double epi_A, double epi_B, double fx, double fy, double dx, double dy,
    Eigen::Matrix<double, 3, 1> light_vec,
    int k_vec_num, int frm_idx,
    Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr,
    double * vertex_set,
    int * vertex_frm,
    std::vector<double*>* parameter_blocks) {

  // Calculate weight
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight
      = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_vec_num, 1);
  double sum_val = 0;
  for (int i = 0; i < k_vec_num; i++) {
    weight(i) = pow((1 - vertex_nbr(i, 1) / vertex_nbr(k_vec_num, 1)), 2);
    sum_val += weight(i);
  }
  weight = weight / sum_val;

  // Set norm_add & depth_add
  Eigen::Matrix<double, 3, 1> norm_add = Eigen::Matrix<double, 3, 1>::Zero();
  double depth_add = 0;
  int k_vec_opt = 0;
  for (int i = 0; i < k_vec_num; i++) {
    int idx = vertex_nbr(i, 0);
    if (vertex_frm[idx] != frm_idx) {
      Eigen::Matrix<double, 3, 1> tmp;
      tmp << vertex_set[idx*4+1], vertex_set[idx*4+2], vertex_set[idx*4+3];
      depth_add += weight(i) * vertex_set[idx*4];
      norm_add += weight(i) * tmp;
    } else {
      k_vec_opt++;
    }
  }

  // Create constraint
  if (k_vec_opt == 0) {
    return nullptr;
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_part;
  weight_part = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_vec_opt, 1);
  int i_opt = 0;
  for (int i = 0; i < k_vec_num; i++) {
    int idx = vertex_nbr(i, 0);
    if (vertex_frm[idx] == frm_idx)
      weight_part(i_opt++) = weight(i);
  }
  DeformConstraint* constraint = new DeformConstraint(
      pattern, img_obs, vec_M, vec_D, epi_A, epi_B, fx, fy, dx, dy, light_vec,
      k_vec_num, k_vec_opt, weight_part, norm_add, depth_add);
  DeformCostFunction* cost_function = new DeformCostFunction(constraint);

  // Add parameter blocks
  parameter_blocks->clear();
  for (int i = 0; i < k_vec_num; i++) {
    int idx = vertex_nbr(i, 0);
    if (vertex_frm[idx] == frm_idx) {
      parameter_blocks->push_back(&(vertex_set[idx*4]));
      cost_function->AddParameterBlock(4);
    }
  }
  cost_function->SetNumResiduals(1);
  return (cost_function);
}