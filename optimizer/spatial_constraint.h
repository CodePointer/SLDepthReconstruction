//
// Created by pointer on 18-4-3.
//

#ifndef SLDEPTHRECONSTRUCTION_SPATIAL_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_SPATIAL_CONSTRAINT_H


#include <ceres/ceres.h>
#include "intensity_slot.h"
#include "regular_constraint.h"

class SpatialConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<SpatialConstraint, 4>
      SpatialCostFunction;

  SpatialConstraint(int k_nbr, std::vector<IntensitySlot*> slots) {
    this->k_nbr_ = k_nbr;
    this->slots_ = std::move(slots);
  }

  template<class T>
  bool operator()(T const* const* pointer_sets, T* residuals) const {
    T diff = T(0);
    T d_k = slots_[0]->GetDepthFromPointer(pointer_sets[0][0]);
    for (int i = 1; i <= k_nbr_; i++) {
      T d_n = slots_[0]->GetDepthFromPointer(pointer_sets[i][0]);
      diff += d_k - d_n;
    }
    residuals[0] = diff / T(k_nbr_);
    return true;
  }

  static SpatialCostFunction* Create(
      int h, int w, IntensitySlot *** slots, cv::Mat * mask, cv::Mat * img_class,
      double * pointer_set, std::vector<double*> * parameter_blocks) {
    // 8 neighbor:
    int k_nbr = 0;
    parameter_blocks->clear();
    int idx = h * kCamWidth + w;
    parameter_blocks->push_back(&pointer_set[idx]);
    std::vector<IntensitySlot *> nbr_slots;

    for (int h_n = h - 1; h_n <= h + 1; h_n++) {
      for (int w_n = w - 1; w_n <= w + 1; w_n++) {
        if (h_n == h && w_n == w) {
          continue;
        }
        if (h_n < 0 || h_n >= kCamHeight || w_n < 0 || w_n >= kCamWidth) {
          continue;
        }
        if (mask->at<uchar>(h_n, w_n) != my::VERIFIED_TRUE) {
          continue;
        }
        k_nbr++;
        int idx_n = h_n * kCamWidth + w_n;
        int c_idx = img_class->at<uchar>(h_n, w_n);
        nbr_slots.push_back(&slots[h_n][w_n][c_idx]);
        parameter_blocks->push_back(&pointer_set[idx_n]);
      }
    }

    SpatialConstraint* constraint = new SpatialConstraint(k_nbr, nbr_slots);
    SpatialCostFunction* cost_function = new SpatialCostFunction(constraint);
    for (int i = 0; i <= k_nbr; ++i) {
      cost_function->AddParameterBlock(1);
    }
    cost_function->SetNumResiduals(1);

    if (k_nbr == 0) {
      return nullptr;
    }
    return cost_function;
  }


  int k_nbr_;
  std::vector<IntensitySlot*> slots_;
};


#endif //SLDEPTHRECONSTRUCTION_SPATIAL_CONSTRAINT_H
