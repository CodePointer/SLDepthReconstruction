//
// Created by pointer on 18-3-9.
//

#include "regular_constraint.h"

RegularConstraint::RegularConstraint(
    int k_nbr, bool flag_in_opt, int k_vec_fix, int k_vec_opt,
    std::vector<double*> vertex_fix,
    Eigen::Matrix<double, Eigen::Dynamic, 1> weight_fix,
    Eigen::Matrix<double, Eigen::Dynamic, 1> weight_opt,
    double alpha_val, double alpha_norm) {
  this->k_nbr_ = k_nbr;
  this->flag_in_opt_ = flag_in_opt;
  this->k_vec_fix_ = k_vec_fix;
  this->k_vec_opt_ = k_vec_opt;
  this->vertex_fix_ = vertex_fix;
  this->weight_fix_ = weight_fix;
  this->weight_opt_ = weight_opt;
  this->alpha_val_ = alpha_val;
  this->alpha_norm_ = alpha_norm;
}

RegularConstraint::RegularCostFunction* RegularConstraint::Create(
    double alpha_val, double alpha_norm, int k_nbr, int frm_idx,
    Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr,
    double * vertex_set,
    int * vertex_frm,
    std::vector<double*>* parameter_blocks
) {
  // Calculate weight
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight
      = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_nbr, 1);
  double sum_val = 0;
  for (int i = 1; i <= k_nbr; i++) {
    weight(i - 1) = pow((1 - vertex_nbr(i, 1)) / vertex_nbr(k_nbr + 1, 1), 2);
    sum_val += weight(i - 1);
  }
  weight = weight / sum_val;

  // Set parameters
  bool flag_in_opt = vertex_frm[(int)vertex_nbr(0, 0)] == frm_idx;
  int k_vec_fix = 0;
  int k_vec_opt = 0;
  for (int i = 0; i <= k_nbr; i++) {
    int idx = (int)vertex_nbr(i, 0);
    if (vertex_frm[idx] == frm_idx) {
      k_vec_opt++;
    } else {
      k_vec_fix++;
    }
  }
  if (k_vec_opt == 0)
    return nullptr;

  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_fix;
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_opt;

  std::vector<double*> vertex_fix;
  vertex_fix.clear();
  parameter_blocks->clear();
  int idx = (int)vertex_nbr(0, 0);
  if (flag_in_opt) {
    parameter_blocks->push_back(&(vertex_set[idx*4]));
    weight_fix = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_vec_fix, 1);
    weight_opt = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_vec_opt - 1, 1);
  } else {
    vertex_fix.push_back(&(vertex_set[idx*4]));
    weight_fix = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_vec_fix - 1, 1);
    weight_opt = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_vec_opt, 1);
  }
  int i_fix = 0;
  int i_opt = 0;
  for (int i = 1; i <= k_nbr; i++) {
    idx = (int)vertex_nbr(i, 0);
    if (vertex_frm[idx] == frm_idx) {
      weight_opt(i_opt++) = weight(i - 1);
      parameter_blocks->push_back(&(vertex_set[idx*4]));
    } else {
      weight_fix(i_fix++) = weight(i - 1);
      vertex_fix.push_back(&(vertex_set[idx*4]));
    }
  }

  RegularConstraint* constraint = new RegularConstraint(
      k_nbr, flag_in_opt, k_vec_fix, k_vec_opt, vertex_fix,
      weight_fix, weight_opt, alpha_val, alpha_norm);
  RegularCostFunction* cost_function = new RegularCostFunction(constraint);

  // Add parameter blocks
  for (int i = 0; i < k_vec_opt; i++) {
    cost_function->AddParameterBlock(4);
  }
  cost_function->SetNumResiduals(1);
  return (cost_function);
}
