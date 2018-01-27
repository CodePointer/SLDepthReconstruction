//
// Created by pointer on 17-12-8.
//

#ifndef DEPTHOPTIMIZATION_REGULAR_COST_FUNCTOR_H
#define DEPTHOPTIMIZATION_REGULAR_COST_FUNCTOR_H

#include "global_fun.h"
#include <ceres/ceres.h>

class RegularCostFunctor {
public:
  RegularCostFunctor(double alpha) : alpha_(alpha) {};

  template <class T>
  bool operator()(const T* const d_k, const T* const d_up,
                  const T* const d_rt, const T* const d_dn,
                  const T* const d_lf, T* residual) const {
    if (ceres::IsNaN(d_k[0])) {
      ErrorThrow("d_k nan error.");
    }
    if (ceres::IsNaN(d_up[0])) {
      ErrorThrow("d_up nan error.");
    }
    if (ceres::IsNaN(d_rt[0])) {
      ErrorThrow("d_rt nan error.");
    }
    if (ceres::IsNaN(d_dn[0])) {
      ErrorThrow("d_dn nan error.");
    }
    if (ceres::IsNaN(d_lf[0])) {
      ErrorThrow("d_lf nan error.");
    }
    int val_up = int(d_up[0] > T(0));
    int val_rt = int(d_rt[0] > T(0));
    int val_dn = int(d_dn[0] > T(0));
    int val_lf = int(d_lf[0] > T(0));
    residual[0] = T(this->alpha_)
                  * (T(4.0) * d_k[0] - T(val_up) * d_up[0] - T(val_dn) * d_dn[0]
                     - T(val_lf) * d_lf[0] - T(val_rt) * d_rt[0])
                  * T(4) / T(val_up + val_rt + val_dn + val_lf);
    if (ceres::IsNaN(residual[0])) {
      ErrorThrow("RegularFunctor, Nan problem");
      std::cout << "Error!" << std::endl;
      return false;
    }
    return true;
  }

  double alpha_;
};


#endif //DEPTHOPTIMIZATION_REGULAR_COST_FUNCTOR_H
