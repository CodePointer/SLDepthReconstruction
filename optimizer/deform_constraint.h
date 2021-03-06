//
// Created by pointer on 18-3-8.
//

#ifndef SLDEPTHRECONSTRUCTION_DEFORM_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_DEFORM_CONSTRAINT_H

#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/cubic_interpolation.h>
#include <vector>
#include <static_para.h>
#include <global_fun.h>

class DeformConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<DeformConstraint, 4>
      DeformCostFunction;

  DeformConstraint(
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
      double img_obs, Eigen::Matrix<double, 3, 1> vec_M,
      Eigen::Matrix<double, 3, 1> vec_D,
      double epi_A, double epi_B, double fx, double fy, double dx, double dy,
      Eigen::Matrix<double, 3, 1> light_vec,
      int k_vec_num, int k_vec_opt,
      Eigen::Matrix<double, Eigen::Dynamic, 1> weight,
      Eigen::Matrix<double, 3, 1> norm_add, double depth_add);

  template <typename T>
  bool operator()(T const* const* vertex_sets, T* residuals) const {
    // Get depth
    T d_k = T(0);
    for (int i = 0; i < k_vec_opt_; i++) {
      d_k += T(weight_(i)) * vertex_sets[i][0];
    }
    d_k += T(depth_add_);
    // Get norm and norm_weight
    Eigen::Matrix<T, 3, 1> norm_vec = Eigen::Matrix<T, 3, 1>::Zero();
    for (int i = 0; i < k_vec_opt_; i++) {
      Eigen::Matrix<T, 3, 1> ver_norm;
      ver_norm << vertex_sets[i][1], vertex_sets[i][2], vertex_sets[i][3];
      ver_norm = ver_norm / ver_norm.norm();
      norm_vec += T(weight_(i)) * ver_norm;
    }
    Eigen::Matrix<T, 3, 1> norm_T_add = norm_add_.cast<T>();
    norm_vec += norm_T_add;
    norm_vec = norm_vec / norm_vec.norm();
    Eigen::Matrix<T, 3, 1> light_vec = light_vec_.cast<T>();
    T norm_weight = light_vec.transpose() * norm_vec;
    if (norm_weight < T(0)) {
      norm_weight = T(0);
    }
    if (norm_weight > T(1) || norm_weight < T(-1)) {
      ErrorThrow("Norm calculation false.");
      return false;
    }
    // Get intensity from pattern
    T img_est_intensity;
    Eigen::Matrix<T, 3, 1> M = this->vec_M_.cast<T>();
    Eigen::Matrix<T, 3, 1> D = this->vec_D_.cast<T>();
    T x_pro = (M(0)*d_k + D(0)) / (M(2)*d_k + D(2));
    T y_pro = T(-this->epi_A_/this->epi_B_) * x_pro + T(1/this->epi_B_);
    this->pattern_.Evaluate(y_pro, x_pro, &img_est_intensity);
    // Set residual
    T img_est = img_est_intensity * norm_weight;
    residuals[0] = img_est - T(img_obs_);
    if (ceres::IsNaN(residuals[0])) {
      ErrorThrow("DeformCostFunction, Nan problem");
      return false;
    }
    return true;
  }

  static DeformCostFunction* Create(
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
      double img_obs, Eigen::Matrix<double, 3, 1> vec_M,
      Eigen::Matrix<double, 3, 1> vec_D,
      double epi_A, double epi_B, double fx, double fy, double dx, double dy,
      Eigen::Matrix<double, 3, 1> light_vec,
      int k_vec_num, int frm_idx,
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr,
      double * vertex_set,
      int * vertex_frm,
      std::vector<double*>* parameter_blocks);

  // pattern & image
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern_;
  double img_obs_;
  // For depth calculation
  Eigen::Matrix<double, 3, 1> vec_M_;
  Eigen::Matrix<double, 3, 1> vec_D_;
  double epi_A_;
  double epi_B_;
  double f_x_, f_y_, d_x_, d_y_;
  Eigen::Matrix<double, 3, 1> light_vec_;

  // For interpolation
  int k_vec_num_;
  int k_vec_opt_;
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_;
  Eigen::Matrix<double, 3, 1> norm_add_;
  double depth_add_;
};


#endif //SLDEPTHRECONSTRUCTION_DEFORM_CONSTRAINT_H
