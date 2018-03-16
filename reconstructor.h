//
// Created by pointer on 17-12-15.
//

#ifndef SLDEPTHRECONSTRUCTION_RECONSTRUCTOR_H
#define SLDEPTHRECONSTRUCTION_RECONSTRUCTOR_H

#include <fstream>
#include <string>
#include <queue>
#include <opencv2/opencv.hpp>
#include <static_para.h>
#include <global_fun.h>
#include "vertex_set.h"
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include "deform_constraint.h"
//#include "deform_cost_functor.h"
#include "regular_constraint.h"
//#include "regular_cost_functor.h"

class Reconstructor {
private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  ////////////////////////////////////////////////////////////////
  /// FileName part. Used for loading.
  ///  main_file_path
  ///  -->dyna_file_path_
  ///  ---->dyna_file_name_, dyna_file_suffix_
  ///  -->pro_file_path_
  ///  ---->pro_file_name, pro_file_suffix_
  ///  -->pattern_file_name, pattern_file_suffix
  ///  -->epi_A_file_name_, epi_B_file_name_
  ///  -->cam_matrix, pro_matrix, rot, trans, light_vec
  ////////////////////////////////////////////////////////////////
  std::string main_file_path_;
  std::string pattern_file_name_;
  std::string pattern_file_suffix_;
  std::string dyna_file_path_;
  std::string dyna_file_name_;
  std::string dyna_file_suffix_;
  std::string pro_file_path_;
  std::string pro_file_name_;
  std::string pro_file_suffix_;
  std::string epi_A_file_name_;
  std::string epi_B_file_name_;
  std::string cam_matrix_name_;
  std::string pro_matrix_name_;
  std::string rots_name_;
  std::string trans_name_;
  std::string light_name_;
//  std::string hard_mask_file_name_;
  ////////////////////////////////////////////////////////////////
  /// Input data & output data part.
  /// Most of them are matrix(cv::Mat).
  /// The calibration part is Eigen::Matrix.
  ////////////////////////////////////////////////////////////////
  CalibSet calib_set_;
  CamMatSet * cam_set_;
  VertexSet * vertex_set_;
  cv::Mat pattern_;
  cv::Mat epi_A_mat_;
  cv::Mat epi_B_mat_;
  cv::Mat hard_mask_; // No value pixels because of calibration
  //TODO: Control Set

  ////////////////////////////////////////////////////////////////
  /// Storage file names.
  ////////////////////////////////////////////////////////////////
  std::string output_file_path_;
  std::string depth_file_path_;
  std::string depth_file_name_;
  std::string vertex_file_path_;
  std::string vertex_file_name_;
  std::string valid_file_path_;
  std::string valid_file_name_;

  ////////////////////////////////////////////////////////////////
  /// Optimization part. Related parameters.
  ////////////////////////////////////////////////////////////////
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> * pat_grid_;

  // Functions:
  bool LoadDatasFromFiles();
  void ConvXpro2Depth(CamMatSet * ptr_cam_set);
  void SetMaskMatFromXpro(int frm_idx);
  void SetMaskMatFromIobs(int frm_idx);
  void SetVertexFromBefore(int frm_idx);
  bool OptimizeVertexSet(int frm_idx);
  bool CalculateDepthMat(int frm_idx);
  bool GenerateIest(int frm_idx);
  bool WriteResult(int frm_idx);

public:
  Reconstructor();
  ~Reconstructor();

  bool Init();
  bool Run();
  bool Close();
};


#endif //SLDEPTHRECONSTRUCTION_RECONSTRUCTOR_H
