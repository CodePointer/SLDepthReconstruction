//
// Created by pointer on 17-12-12.
//

#ifndef DEPTHOPTIMIZATION_DEFORM_COST_FUNCTOR_H
#define DEPTHOPTIMIZATION_DEFORM_COST_FUNCTOR_H


#include <ceres/cubic_interpolation.h>
#include <ceres/ceres.h>
#include <static_para.h>
#include "global_fun.h"

class DeformCostFunctor {
public:
  DeformCostFunctor(
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern,
      double img_obs, double img_obs_last, double img_est_last,
      Eigen::Matrix<double, 3, 1> vec_M,
      Eigen::Matrix<double, 3, 1> vec_D, double epi_A, double epi_B,
      Eigen::Matrix<double, Eigen::Dynamic, 1> weight,
      double fx, double fy, double dx, double dy,
      Eigen::Matrix<double, 3, 1> light_vec);

  template <class T>
  bool operator()(const T* const v_1, const T* const v_2,
                  const T* const v_3, const T* const v_4, T* residuals) const {
    if (ceres::IsNaN(v_1[0])) {
      ErrorThrow("v_ul nan error.");
    }
    if (ceres::IsNaN(v_2[0])) {
      ErrorThrow("v_ur nan error.");
    }
    if (ceres::IsNaN(v_3[0])) {
      ErrorThrow("v_dl nan error.");
    }
    if (ceres::IsNaN(v_4[0])) {
      ErrorThrow("v_dr nan error.");
    }
    // get d_k
    double d_k = 0;
    d_k += weight_(0) * v_1[0];
    d_k += weight_(1) * v_2[0];
    d_k += weight_(2) * v_3[0];
    d_k += weight_(3) * v_4[0];

    // Get x_pro, y_pro, Intensity from pattern
    Eigen::Matrix<T, 3, 1> M = this->vec_M_.cast<T>();
    Eigen::Matrix<T, 3, 1> D = this->vec_D_.cast<T>();
    T x_pro = (M(0)*d_k + D(0)) / (M(2)*d_k + D(2));
    T y_pro = T(-this->epi_A_/this->epi_B_) * x_pro + T(1/this->epi_B_);
    T img_est_intensity;
    this->pattern_.Evaluate(y_pro, x_pro, &img_est_intensity);

    // Get norm
    Eigen::Matrix<T, 3, 1> tmp_ul, tmp_ur, tmp_dl, tmp_dr;
    Eigen::Matrix<T, 3, 1> obj_ul, obj_ur, obj_dl, obj_dr;
    Eigen::Matrix<T, 3, 1> vec_ul2dr, vec_dl2ur;
    tmp_ul << T((range_(0, 1) - d_x_) / f_x_), T((range_(0, 0) - d_y_) / f_y_), T(1.0);
    tmp_ur << T((range_(1, 1) - d_x_) / f_x_), T((range_(0, 0) - d_y_) / f_y_), T(1.0);
    tmp_dl << T((range_(0, 1) - d_x_) / f_x_), T((range_(1, 0) - d_y_) / f_y_), T(1.0);
    tmp_dr << T((range_(1, 1) - d_x_) / f_x_), T((range_(1, 0) - d_y_) / f_y_), T(1.0);
    obj_ul = v_ul[0] * tmp_ul;
    obj_ur = v_ur[0] * tmp_ur;
    obj_dl = v_dl[0] * tmp_dl;
    obj_dr = v_dr[0] * tmp_dr;
    vec_ul2dr = obj_dr - obj_ul;
    vec_dl2ur = obj_ur - obj_dl;
    Eigen::Matrix<T, 3, 1> norm_vec, light_vec;
    light_vec = light_vec_.cast<T>();
    norm_vec = vec_ul2dr.cross(vec_dl2ur);
    norm_vec = norm_vec / norm_vec.norm();
    T norm_weight = light_vec.transpose() * norm_vec;

    T img_est = img_est_intensity * norm_weight;

    residuals[0] = img_est - T(img_obs_);
//    residuals[0] = T(img_obs_ / img_obs_last_) - img_est / T(img_est_last_);
    if (ceres::IsNaN(residuals[0])) {
      ErrorThrow("DeformFunctor, Nan problem");
      std::cout << "Error!" << std::endl;
      return false;
    }
    return true;
  }

  // pattern & weight
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & pattern_;
  // ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> & weight_;
  // For depth color calculation
  double img_obs_;
  double img_obs_last_;
  double img_est_last_;
  Eigen::Matrix<double, 3, 1> vec_M_;
  Eigen::Matrix<double, 3, 1> vec_D_;
  double epi_A_;
  double epi_B_;
  // For interpolation
  Eigen::Matrix<double, Eigen::Dynamic, 1> weight_;

  double f_x_, f_y_, d_x_, d_y_;
  Eigen::Matrix<double, 3, 1> light_vec_;
};


#endif //DEPTHOPTIMIZATION_DEFORM_COST_FUNCTOR_H
