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

  RegularConstraint(int k_nbr, bool flag_in_opt, int k_vec_fix, int k_vec_opt,
                    std::vector<double*> vertex_fix,
                    Eigen::Matrix<double, Eigen::Dynamic, 1> weight_fix,
                    Eigen::Matrix<double, Eigen::Dynamic, 1> weight_opt,
                    double alpha_val, double alpha_norm);

  template <class T>
  bool operator()(T const* const* vertex_sets, T* residuals) const {
    // Calculate gradient on value
    T grad_val = T(0);
    if (flag_in_opt_) {
      for (int i = 1; i < k_vec_opt_; i++) {
        T diff = vertex_sets[0][0] - vertex_sets[i][0];
        grad_val += T(weight_opt_(i - 1)) * diff;
      }
      for (int i = 0; i < k_vec_fix_; i++) {
        T diff = vertex_sets[0][0] - T(vertex_fix_[i][0]);
        grad_val += T(weight_fix_(i)) * diff;
      }
    } else {
      for (int i = 0; i < k_vec_opt_; i++) {
        T diff = T(vertex_fix_[0][0]) - vertex_sets[i][0];
        grad_val += T(weight_opt_(i)) * diff;
      }
      for (int i = 1; i < k_vec_fix_; i++) {
        T diff = T(vertex_fix_[0][0]) - T(vertex_fix_[i][0]);
        grad_val += T(weight_fix_(i - 1)) * diff;
      }
    }

    // Calculate gradient on norm
    T grad_norm = T(0);
    if (flag_in_opt_) {
      Eigen::Matrix<T, 3, 1> vertex_norm;
      vertex_norm << vertex_sets[0][1], vertex_sets[0][2], vertex_sets[0][3];
      vertex_norm = vertex_norm / vertex_norm.norm();
      for (int i = 1; i < k_vec_opt_; i++) {
        Eigen::Matrix<T, 3, 1> vertex_nbr_norm;
        vertex_nbr_norm
            << vertex_sets[i][1], vertex_sets[i][2], vertex_sets[i][3];
        vertex_nbr_norm = vertex_nbr_norm / vertex_nbr_norm.norm();
        grad_norm +=
            T(weight_opt_(i - 1)) * vertex_nbr_norm.transpose() * vertex_norm;
      }
      for (int i = 0; i < k_vec_fix_; i++) {
        Eigen::Matrix<T, 3, 1> vertex_nbr_norm;
        vertex_nbr_norm << T(vertex_fix_[i][1]), T(vertex_fix_[i][2]),
            T(vertex_fix_[i][3]);
        vertex_nbr_norm = vertex_nbr_norm / vertex_nbr_norm.norm();
        grad_norm +=
            T(weight_fix_(i)) * vertex_nbr_norm.transpose() * vertex_norm;
      }
    } else {
      Eigen::Matrix<T, 3, 1> vertex_norm;
      vertex_norm << T(vertex_fix_[0][1]), T(vertex_fix_[0][2]),
          T(vertex_fix_[0][3]);
      vertex_norm = vertex_norm / vertex_norm.norm();
      for (int i = 0; i < k_vec_opt_; i++) {
        Eigen::Matrix<T, 3, 1> vertex_nbr_norm;
        vertex_nbr_norm
            << vertex_sets[i][1], vertex_sets[i][2], vertex_sets[i][3];
        vertex_nbr_norm = vertex_nbr_norm / vertex_nbr_norm.norm();
        grad_norm += T(weight_opt_(i))
                     * vertex_nbr_norm.transpose() * vertex_norm;
      }
      for (int i = 1; i < k_vec_fix_; i++) {
        Eigen::Matrix<T, 3, 1> vertex_nbr_norm;
        vertex_nbr_norm << T(vertex_fix_[i][1]), T(vertex_fix_[i][2]),
            T(vertex_fix_[i][3]);
        vertex_nbr_norm = vertex_nbr_norm / vertex_nbr_norm.norm();
        grad_norm +=
            T(weight_fix_(i - 1)) * vertex_nbr_norm.transpose() * vertex_norm;
      }
    }
    residuals[0] = T(alpha_val_) * grad_val + T(alpha_norm_) * grad_norm;
    if (ceres::IsNaN(residuals[0])) {
      ErrorThrow("RegularConstraint, Nan problem");
      return false;
    }
    return true;
  }

  static RegularCostFunction* Create(
      double alpha_val, double alpha_norm, int k_nbr, int frm_idx,
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr,
      double * vertex_set,
      int * vertex_frm,
      std::vector<double*>* parameter_blocks
  );

  int k_nbr_;
  bool flag_in_opt_;
  int k_vec_fix_;
  int k_vec_opt_;
  std::vector<double*> vertex_fix_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_fix_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_opt_;

  double alpha_val_;
  double alpha_norm_;
};


#endif //SLDEPTHRECONSTRUCTION_REGULAR_CONSTRAINT_H
