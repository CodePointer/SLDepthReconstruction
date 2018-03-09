//
// Created by pointer on 18-3-9.
//

#include "regular_constraint.h"

RegularConstraint::RegularConstraint(
    double alpha_val, double alpha_norm, int k_nbr,
    Eigen::Matrix<double, Eigen::Dynamic, 1> weight) {
  this->alpha_val_ = alpha_val;
  this->alpha_norm_ = alpha_norm;
  this->k_nbr_ = k_nbr;
  this->weight_ = weight;
}

RegularConstraint::RegularCostFunction* RegularConstraint::Create(
    double alpha_val, double alpha_norm, int k_nbr,
    Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr,
    double * vertex_set,
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

  RegularConstraint* constraint = new RegularConstraint(
      alpha_val, alpha_norm, k_nbr, weight);
  RegularCostFunction* cost_function = new RegularCostFunction(constraint);
  // Add parameter blocks
  parameter_blocks->clear();
  for (int i = 0; i <= k_nbr; i++) {
    int idx = (int)vertex_nbr(i, 0);
    parameter_blocks->push_back(&(vertex_set[idx*4]));
    cost_function->AddParameterBlock(4);
  }
  cost_function->SetNumResiduals(1);
  return (cost_function);
}
