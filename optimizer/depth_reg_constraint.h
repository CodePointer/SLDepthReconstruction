//
// Created by pointer on 18-4-18.
//

#ifndef SLDEPTHRECONSTRUCTION_DEPTH_REG_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_DEPTH_REG_CONSTRAINT_H

#include <ceres/ceres.h>
#include "intensity_slot.h"

class DepthRegConstraint {
public:
  typedef ceres::AutoDiffCostFunction<DepthRegConstraint, 1, 1, 1>
      DepthRegCostFunction;

  DepthRegConstraint(double weight, IntensitySlot * slot_a,
                     IntensitySlot * slot_b, std::string info){
    this->weight_ = weight;
    this->slot_a_ = slot_a;
    this->slot_b_ = slot_b;
    this->info_ = info;
  }

  template <class T>
  bool operator()(const T* const pointer_a,
                  const T* const pointer_b, T* residual) const {
    if (slot_a_ == nullptr || slot_b_ == nullptr) {
      LOG(ERROR) << "Nullptr:" << info_;
    }
    try {
      T depth_a = slot_a_->GetDepthFromPointer<T>(pointer_a[0]);
      T depth_b = slot_b_->GetDepthFromPointer<T>(pointer_b[0]);
//    residual[0] = T(0.0);
      residual[0] = T(weight_) * (depth_a - depth_b);
      return true;
    } catch (std::exception e) {
      LOG(ERROR) << info_ << std::endl;
      std::cerr << info_ << std::endl;
    }
  }

  static DepthRegCostFunction* Create(double dist_s, double dist_t,
                                      double omega_s, double omega_t,
                                      IntensitySlot * slot_a,
                                      IntensitySlot * slot_b, std::string info) {
    double weight_s = 1 / (sqrt(2*M_PI) * omega_s)
                      * exp(-(dist_s*dist_s)/(2*omega_s*omega_s));
    double weight_t = 1 / (sqrt(2*M_PI) * omega_t)
                      * exp(-(dist_t*dist_t)/(2*omega_t*omega_t));
    return new DepthRegCostFunction(
        new DepthRegConstraint(weight_s * weight_t * 1e4, slot_a, slot_b, info));
  }

  double weight_;
  IntensitySlot * slot_a_;
  IntensitySlot * slot_b_;
  std::string info_;
};


#endif //SLDEPTHRECONSTRUCTION_DEPTH_REG_CONSTRAINT_H