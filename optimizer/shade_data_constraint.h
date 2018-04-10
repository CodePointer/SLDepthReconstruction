//
// Created by pointer on 18-4-4.
//

#ifndef SLDEPTHRECONSTRUCTION_SHADING_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_SHADING_CONSTRAINT_H


#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

class ShadeDataConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<ShadeDataConstraint, 4>
      ShadeDataCostFunction;

  ShadeDataConstraint(double img_obs_k, double img_pattern_k, int k_vec_num,
                      Eigen::Matrix<double, Eigen::Dynamic, 1> weight) {
    this->img_obs_k_ = img_obs_k;
    this->img_pattern_k_ = img_pattern_k;
    this->k_vec_num_ = k_vec_num;
    this->weight_ = std::move(weight);
  }

  template <class T>
  bool operator()(T const* const* vertex_sets, T* residuals) const {
    // Get shade
    T weight_shade = T(0);
    for (int i = 0; i < k_vec_num_; i++) {
      weight_shade += vertex_sets[i][0] * T(weight_(i));
    }
    residuals[0] = T(img_obs_k_) - T(img_pattern_k_) * weight_shade;
    return true;
  }

  static ShadeDataCostFunction* Create(
      double img_obs_k, double img_pattern_k, int k_vec_num,
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr,
      double * vertex_set, std::vector<double*>* parameter_blocks) {
    // Calculate weight
    Eigen::Matrix<double, Eigen::Dynamic, 1> weight
        = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(k_vec_num, 1);
    double sum_val = 0;
    for (int i = 0; i < k_vec_num; i++) {
      weight(i) = pow((1 - vertex_nbr(i, 1) / vertex_nbr(k_vec_num, 1)), 2);
      sum_val += weight(i);
    }
    weight = weight / sum_val;

    // Add vertex_sets (only the gray part)
    ShadeDataConstraint* constraint = new ShadeDataConstraint(
        img_obs_k, img_pattern_k, k_vec_num, weight);
    ShadeDataCostFunction* cost_function = new ShadeDataCostFunction(constraint);
    parameter_blocks->clear();
    for (int i = 0; i < k_vec_num; i++) {
      int idx = (int)vertex_nbr(i, 0);
      parameter_blocks->push_back(&vertex_set[idx*4]);
      cost_function->AddParameterBlock(1);
    }
    cost_function->SetNumResiduals(1);
    return cost_function;
  }

  double img_obs_k_;
  double img_pattern_k_;
  int k_vec_num_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_;
};


#endif //SLDEPTHRECONSTRUCTION_SHADING_CONSTRAINT_H
