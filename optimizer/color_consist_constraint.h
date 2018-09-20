//
// Created by pointer on 18-9-18.
//

#ifndef SLDEPTHRECONSTRUCTION_COLOR_CONSIST_CONSTRAINT_H
#define SLDEPTHRECONSTRUCTION_COLOR_CONSIST_CONSTRAINT_H


#include <ceres/ceres.h>
#include <static_para.h>

#include <utility>
#include <node_set.h>

class ColorConsistConstraint {
public:
  typedef ceres::DynamicAutoDiffCostFunction<ColorConsistConstraint, 4>
    ColorConsistCostFunction;

  ColorConsistConstraint(std::string info, double u, double v, int img_class,
                         Eigen::Vector3d vec_M, Eigen::Vector3d vec_D) {
    info_ = std::move(info);
    u_ = u;
    v_ = v;
    img_class_ = img_class;
    vec_M_ = vec_M;
    vec_D_ = vec_D;
  }

  template <class T>
  bool operator()(T const* const* depth_vals, T* residuals) const {
    T depth_val = (T(1) - T(u_) - T(v_)) * depth_vals[0][0]
                       + T(u_) * depth_vals[1][0] + T(v_) * depth_vals[2][0];
    T x_pro_val = (T(vec_M_(0))*depth_val + T(vec_D_(0)))
                  / (T(vec_M_(2))*depth_val + T(vec_D_(2)));
    // Check the img class
    double x_pro_center = (7 - img_class_) * kClassNum;
    T period_dis = ceres::floor((x_pro_val - T(x_pro_center))
                                / T(kClassNum * kStripDis))
                   * T(kClassNum * kStripDis);
    if (ceres::abs(x_pro_val - period_dis - T(x_pro_center)) < T(6)) {
      residuals[0] = T(0);
    } else {
      residuals[0] = T(0.5) * (ceres::cos(T(3.1415926)*x_pro_val/T(6)) + T(1));
    }

    return !ceres::IsNaN(residuals[0]);
  }

  static ColorConsistCostFunction* Create(
      std::string info, NodeSet* node_set, CalibSet* p_calib,
      CamMatSet * cam_set, int h_cam, int w_cam, std::vector<int>* idx_list,
      std::vector<double*>* parameter_blocks) {
    ColorConsistConstraint* constraint = nullptr;
    idx_list->clear();
    // Find 3 vertex
    int idx_mesh = cam_set->mesh_mat.at<ushort>(h_cam, w_cam);
    cv::Point3i vertex_set = node_set->mesh_[idx_mesh];
    if (vertex_set.x < 0 || vertex_set.y < 0 || vertex_set.z < 0
        || vertex_set.x >= node_set->len_ || vertex_set.y >= node_set->len_
        || vertex_set.z >= node_set->len_) {
      return nullptr;
    }
    if (vertex_set.x == vertex_set.y || vertex_set.y == vertex_set.z || vertex_set.x == vertex_set.z) {
      return nullptr;
    }
    idx_list->push_back(vertex_set.x);
    idx_list->push_back(vertex_set.y);
    idx_list->push_back(vertex_set.z);
    // Get vec_M, vec_D
    int idx = h_cam * kCamWidth + w_cam;
    Eigen::Vector3d vec_M = p_calib->M.block<3, 1>(0, idx);
    Eigen::Vector3d vec_D = p_calib->D;
    // Get u, v
    double u = cam_set->uv_weight(0, h_cam * kCamWidth + w_cam);
    double v = cam_set->uv_weight(1, h_cam * kCamWidth + w_cam);
    // check valid
    if (u >= 0 && u <= 1 && v >= 0 && v <= 1 && u+v >= 0 && u+v <= 1) {
      constraint = new ColorConsistConstraint(
          info, u, v, cam_set->img_class.at<uchar>(h_cam, w_cam), vec_M, vec_D);
      ColorConsistCostFunction * cost_function
          = new ColorConsistCostFunction(constraint);
      parameter_blocks->clear();

      for (int i = 0; i < 3; i++) {
        int idx_i = (*idx_list)[i];
        parameter_blocks->push_back(&(node_set->val_.data()[idx_i]));
        cost_function->AddParameterBlock(1);
      }
      cost_function->SetNumResiduals(1);
      return cost_function;
    } else {
      return nullptr;
    }
  }

  std::string info_;
  double u_;
  double v_;
  int img_class_;
  Eigen::Vector3d vec_M_;
  Eigen::Vector3d vec_D_;
};


#endif //SLDEPTHRECONSTRUCTION_COLOR_CONSIST_CONSTRAINT_H
