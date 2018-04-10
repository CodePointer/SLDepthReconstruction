//
// Created by pointer on 18-4-3.
//

#ifndef SLDEPTHRECONSTRUCTION_TEMPORAL_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_TEMPORAL_CONSTRAINT_H


#include <ceres/ceres.h>
#include <opencv2/core/mat.hpp>
#include <utility>
#include <intensity_slot.h>

class TemporalConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<TemporalConstraint, 4>
      TemporalCostFunction;

  TemporalConstraint(double weight, int k_his, std::vector<double> d_set,
                     IntensitySlot * slot, cv::Mat * img_class) {
    this->weight_ = weight;
    this->k_his_ = k_his;
    this->d_set_ = std::move(d_set);
    this->slot_ = slot;
    this->img_class_ = img_class;
  }

  template<class T>
  bool operator()(T const* const* pointer, T* residuals) const {
    T diff = T(0);
    T d_now = slot_->GetDepthFromPointer(pointer[0][0]);
    for (int i = 0; i < k_his_; i++) {
      diff += d_now - T(d_set_[i]);
    }
    residuals[0] = T(weight_) / T(k_his_) * diff;
    return true;
  }

  static TemporalCostFunction* Create(int h, int w, double weight, int k_his,
                                      CamMatSet * cam_set, int frm_idx,
                                      IntensitySlot *** slots, double * pointer,
                                      std::vector<double*> * parameter_blocks) {
    if (k_his <= 0) {
      return nullptr;
    }
    parameter_blocks->clear();
    parameter_blocks->push_back(pointer);
    std::vector<double> d_set;
    for (int i = 1; i <= k_his; i++) {
      d_set.push_back(cam_set[frm_idx - i].depth.at<double>(h, w));
    }

    TemporalConstraint* constraint
        = new TemporalConstraint(weight, k_his, d_set, slots[h][w],
                                 &cam_set[frm_idx].img_class);
    TemporalCostFunction* cost_function = new TemporalCostFunction(constraint);
    cost_function->AddParameterBlock(1);
    cost_function->SetNumResiduals(1);

    return cost_function;
  }

  double weight_;
  int k_his_;
  std::vector<double> d_set_;
  IntensitySlot * slot_;
  cv::Mat * img_class_;
};


#endif //SLDEPTHRECONSTRUCTION_TEMPORAL_CONSTRAINT_H
