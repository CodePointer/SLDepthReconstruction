//
// Created by pointer on 18-4-4.
//

#ifndef SLDEPTHRECONSTRUCTION_SHADE_REG_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_SHADE_REG_CONSTRAINT_H


#include <ceres/ceres.h>

class ShadeRegConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<ShadeRegConstraint, 4>
      ShadeRegCostFunction;

  ShadeRegConstraint(int k_nbr, double weight_reg,
                     Eigen::Matrix<double, Eigen::Dynamic, 1> weight) {
    this->k_nbr_ = k_nbr;
    this->weight_reg_ = weight_reg;
    this->weight_ = weight;
  }

  template <class T>
  bool operator()(T const* const* vertex_sets, T* residuals) const {
    T grad_diff = T(0);
    for (int i = 1; i <= k_nbr_; i++) {
      T diff = vertex_sets[0][0] - vertex_sets[i][0];
      grad_diff += T(weight_(i-1)) * diff;
    }
    residuals[0] = T(weight_reg_) * grad_diff;
    if (ceres::IsNaN(residuals[0])) {
      LOG(ERROR) << "residual Nan.";
      return false;
    }
    return true;
  }

  // vertex_nbr[k_nbr+2] = [self, nbr1, ... nbr(k), nbr(k+1)];
  static ShadeRegCostFunction* Create(
      int k_nbr, double weight_reg,
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr,
      double* vertex_set, std::vector<double*>* parameter_blocks) {
    // Calculate weight
    Eigen::Matrix<double, Eigen::Dynamic, 1> weight
        = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_nbr, 1);
    double sum_val = 0;
    for (int i = 1; i <= k_nbr; i++) {
      weight(i) = pow((1 - vertex_nbr(i, 1) / vertex_nbr(k_nbr + 1, 1)), 2);
      sum_val += weight(i);
    }
    weight = weight / sum_val;
    // Add
    ShadeRegConstraint* constraint = new ShadeRegConstraint(
        k_nbr, weight_reg, weight);
    ShadeRegCostFunction* cost_function = new ShadeRegCostFunction(constraint);
    parameter_blocks->clear();
    for (int i = 0; i <= k_nbr; i++) {
      int idx = (int)vertex_nbr(i, 0);
      parameter_blocks->push_back(&vertex_set[idx*4]);
      cost_function->AddParameterBlock(1);
    }
    cost_function->SetNumResiduals(1);
    return cost_function;
  }

  int k_nbr_;
  double weight_reg_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_;
};


#endif //SLDEPTHRECONSTRUCTION_SHADE_REG_CONSTRAINT_H
