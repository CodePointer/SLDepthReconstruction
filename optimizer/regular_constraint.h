//
// Created by pointer on 18-3-9.
//

#ifndef SLDEPTHRECONSTRUCTION_REGULAR_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_REGULAR_CONSTRAINT_H


#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <vector>
#include <static_para.h>
#include <global_fun.h>

class RegularConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<RegularConstraint, 4>
      RegularCostFunction;

  RegularConstraint(double alpha_val, double alpha_norm, int k_nbr,
                    Eigen::Matrix<double, Eigen::Dynamic, 1> weight);

  template <class T>
  bool operator()(T const* const* vertex_sets, T* residuals) const {
    // Calculate gradient on value
    T grad_val = T(0);
    for (int i = 1; i <= k_nbr_; i++) {
      T diff = vertex_sets[0][0] - vertex_sets[i][0];
      grad_val += T(weight_(i - 1)) * diff;
    }
    // Calculate gradient on norm
    T grad_norm = T(0);
    Eigen::Matrix<T, 3, 1> vertex_norm;
    vertex_norm << vertex_sets[0][1], vertex_sets[0][2], vertex_sets[0][3];
    for (int i = 1; i <= k_nbr_; i++) {
      Eigen::Matrix<T, 3, 1> vertex_nbr_norm;
      vertex_nbr_norm << vertex_sets[i][1], vertex_sets[i][2], vertex_sets[i][3];
      grad_norm += T(weight_(i - 1)) * vertex_norm.transpose() * vertex_nbr_norm;
    }
    residuals[0] = T(alpha_val_) * grad_val + T(alpha_norm_) * grad_norm;
    if (ceres::IsNaN(residuals[0])) {
      ErrorThrow("RegularConstraint, Nan problem");
      return false;
    }
    return true;
  }

  static RegularCostFunction* Create(
      double alpha_val, double alpha_norm, int k_nbr,
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr,
      double * vertex_set,
      std::vector<double*>* parameter_blocks
  );


  int k_nbr_;
  double alpha_val_;
  double alpha_norm_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_;
};


#endif //SLDEPTHRECONSTRUCTION_REGULAR_CONSTRAINT_H
