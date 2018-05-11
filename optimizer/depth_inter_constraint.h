//
// Created by pointer on 18-5-8.
//

#ifndef SLDEPTHRECONSTRUCTION_DEPTH_INTER_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_DEPTH_INTER_CONSTRAINT_H

#include <ceres/ceres.h>

#include <utility>
#include "static_para.h"
#include "node_set.h"

class DepthInterConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<DepthInterConstraint, 4>
      DepthInterCostFunction;

  DepthInterConstraint(std::string info, bool flag_fix,
                       double * fix_pointer) {
    this->info_ = std::move(info);
    this->flag_fix_ = flag_fix;
    if (flag_fix) {
      this->fix_val_ = *fix_pointer;
    } else {
      this->fix_val_ = -1.0;
    }
  }

  template <class T>
  bool operator()(T const* const* depth_vals, T* residuals) const {
    if (flag_fix_) {
      residuals[0] = *(depth_vals[0]) - T(fix_val_);
    } else {
      residuals[0] = *(depth_vals[0]) - *(depth_vals[1]);
    }
    return true;
  }

  static DepthInterCostFunction* Create(
      std::string info, bool flag_fix,
      double* depth_pointer, double* nbr_pointer,
      std::vector<double*>* parameter_blocks) {
    // Create constraints
    DepthInterConstraint* constraint = nullptr;
    if (flag_fix) {
      constraint = new DepthInterConstraint(std::move(info), flag_fix, nbr_pointer);
    } else {
      constraint = new DepthInterConstraint(std::move(info), flag_fix, nullptr);
    }
    DepthInterCostFunction* cost_function = new DepthInterCostFunction(constraint);

    // Set parameters
    parameter_blocks->clear();
    parameter_blocks->push_back(depth_pointer);
    cost_function->AddParameterBlock(1);
    if (!flag_fix) {
      parameter_blocks->push_back(nbr_pointer);
      cost_function->AddParameterBlock(1);
    }
    cost_function->SetNumResiduals(1);
    return (cost_function);
  }

  std::string info_;
  bool flag_fix_;
  double fix_val_;
};


#endif //SLDEPTHRECONSTRUCTION_DEPTH_INTER_CONSTRAINT_H
