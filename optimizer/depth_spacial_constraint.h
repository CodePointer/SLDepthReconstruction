//
// Created by pointer on 18-6-30.
//

#ifndef SLDEPTHRECONSTRUCTION_DEPTHSPATIALCONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_DEPTHSPATIALCONSTRAINT_H

#include <ceres/ceres.h>
#include <utility>
#include "static_para.h"
#include "node_set.h"

class DepthSpacialConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<DepthSpacialConstraint, 4>
      DepthSpacialCostFunction;

  DepthSpacialConstraint(std::string info, int up_num, int dn_num) {
    info_ = info;
    up_num_ = up_num;
    dn_num_ = dn_num;
  }

  template<class T>
  bool operator()(T const* const* depth_vals, T* residuals) const {
    T up_value = T(0);
    T dn_value = T(0);
    for (int i = 1; i <= up_num_; i++) {
      up_value += depth_vals[i][0];
    }
    up_value = up_value / T(up_num_);
    for (int i = up_num_ + 1; i <= up_num_ + dn_num_; i++) {
      dn_value += depth_vals[i][0];
    }
    dn_value = dn_value / T(dn_num_);
    residuals[0] = up_value + dn_value  - T(2) * depth_vals[0][0];
    return true;
  }

  static DepthSpacialCostFunction* Create(
      std::string info, NodeSet* node_set, int idx_i, bool ver_flag,
      std::vector<int>* idx_list,
      std::vector<double*>* parameter_blocks) {
    // Create constraints
    DepthSpacialConstraint* constraint = nullptr;
    idx_list->clear();
    idx_list->push_back(idx_i);
    // Find neighbor
    int h_cen, w_cen;
    node_set->GetNodeCoordByIdx(idx_i, &h_cen, &w_cen);
    int pos_x = node_set->pos_(idx_i, 0);
    int pos_y = node_set->pos_(idx_i, 1);
    int up_num = 0, dn_num = 0;
    std::vector<double*> up_value; up_value.clear();
    std::vector<double*> dn_value; dn_value.clear();
    if (ver_flag) {
      for (int nw = w_cen - 1; nw <= w_cen + 1; nw++) {
        int idx_up = node_set->GetIdxByNodeCoord(h_cen - 1, nw, true);
        if (idx_up > 0) {
          up_num++; up_value.push_back(&(node_set->val_.data()[idx_up]));
          idx_list->push_back(idx_up);
        }
        int idx_dn = node_set->GetIdxByNodeCoord(h_cen + 1, nw, true);
        if (idx_dn > 0) {
          dn_num++; dn_value.push_back(&(node_set->val_.data()[idx_dn]));
          idx_list->push_back(idx_dn);
        }
      }
    } else {
      for (int nh = h_cen - 1; nh <= h_cen + 1; nh++) {
        int idx_lf = node_set->GetIdxByNodeCoord(nh, w_cen - 1, true);
        if (idx_lf > 0) {
          up_num++; up_value.push_back(&(node_set->val_.data()[idx_lf]));
          idx_list->push_back(idx_lf);
        }
        int idx_rt = node_set->GetIdxByNodeCoord(nh, w_cen + 1, true);
        if (idx_rt > 0) {
          dn_num++; dn_value.push_back(&(node_set->val_.data()[idx_rt]));
          idx_list->push_back(idx_rt);
        }
      }
    }
    if (up_num == 0 || dn_num == 0)
      return nullptr;
    // Set parameters
    constraint = new DepthSpacialConstraint(info, up_num, dn_num);
    DepthSpacialCostFunction* cost_function
        = new DepthSpacialCostFunction(constraint);
    parameter_blocks->clear();
    parameter_blocks->push_back(&(node_set->val_.data()[idx_i]));
    cost_function->AddParameterBlock(1);
    for (int i = 0; i < up_num; i++) {
      parameter_blocks->push_back(up_value[i]);
      cost_function->AddParameterBlock(1);
    }
    for (int i = 0; i < dn_num; i++) {
      parameter_blocks->push_back(dn_value[i]);
      cost_function->AddParameterBlock(1);
    }
    cost_function->SetNumResiduals(1);
    return cost_function;
  }

  std::string info_;
  int up_num_;
  int dn_num_;
};


#endif //SLDEPTHRECONSTRUCTION_DEPTHSPATIALCONSTRAINT_H
