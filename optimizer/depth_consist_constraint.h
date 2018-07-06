//
// Created by pointer on 18-6-30.
//

#ifndef SLDEPTHRECONSTRUCTION_DEPTH_CONTI_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_DEPTH_CONTI_CONSTRAINT_H


#include <ceres/ceres.h>
#include <node_set.h>

class DepthConsistConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<DepthConsistConstraint>
      DepthCosistCostFunction;

  DepthConsistConstraint(std::string info, double depth_start) {
    info_ = info;
    depth_start_ = depth_start;
  }

  template <class T>
  bool operator()(T const* const* depth_vals, T* residuals) const {
    residuals[0] = (depth_vals[0][0] - depth_vals[1][0])
                   - (depth_vals[1][0] - T(depth_start_));
    return true;
  }

  static DepthCosistCostFunction* Create(
      std::string info, int idx_i, NodeSet* node_set_t,
      NodeSet* node_set_t1, NodeSet* node_set_t2,
      std::vector<double*>* parameter_blocks) {
    // Create constraints
    DepthConsistConstraint* constraint = nullptr;
    // Check valid
    if (node_set_t->valid_(idx_i, 0) == my::VERIFIED_TRUE
        && node_set_t1->valid_(idx_i, 0) == my::VERIFIED_TRUE
        && node_set_t2->valid_(idx_i, 0) == my::VERIFIED_TRUE) {
      constraint = new DepthConsistConstraint(info, node_set_t2->val_(idx_i, 0));
      DepthCosistCostFunction* cost_function
          = new DepthCosistCostFunction(constraint);
      parameter_blocks->clear();
      parameter_blocks->push_back(&(node_set_t->val_.data()[idx_i]));
      cost_function->AddParameterBlock(1);
      parameter_blocks->push_back(&(node_set_t1->val_.data()[idx_i]));
      cost_function->AddParameterBlock(1);
      cost_function->SetNumResiduals(1);
      return cost_function;
    } else {
      return nullptr;
    }
  }

  std::string info_;
  double depth_start_;
};


#endif //SLDEPTHRECONSTRUCTION_DEPTH_CONTI_CONSTRAINT_H
