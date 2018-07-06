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
#include <sys/time.h>
#include <node_set.h>
#include "intensity_slot.h"
#include "depth_reg_constraint.h"
#include "shade_data_constraint.h"
#include "depth_inter_constraint.h"

static timeval g_time_last;

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
  int kFrameNum;
  std::string main_file_path_;
//  std::string pattern_file_name_;
//  std::string class_file_name_;
//  std::string pattern_file_suffix_;
  std::string dyna_file_path_;
  std::string dyna_file_name_;
  std::string dyna_file_suffix_;
  std::string pro_file_path_;
  std::string pro_file_name_;
  std::string pro_file_suffix_;
//  std::string epi_A_file_name_;
//  std::string epi_B_file_name_;
  std::string cam_matrix_name_;
  std::string pro_matrix_name_;
  std::string rots_name_;
  std::string trans_name_;
//  std::string light_name_;
//  std::string hard_mask_name_;
//  std::string hard_mask_file_name_;
  ////////////////////////////////////////////////////////////////
  /// Input data & output data part.
  /// Most of them are matrix(cv::Mat).
  /// The calibration part is Eigen::Matrix.
  ////////////////////////////////////////////////////////////////
  CalibSet calib_set_;
  CamMatSet * cam_set_;
  NodeSet * node_set_;
//  CamSlotsMat * cam_slots_;
//  VertexSet * vertex_set_;
//  cv::Mat pattern_;
//  cv::Mat pattern_class_;
//  cv::Mat epi_A_mat_;
//  cv::Mat epi_B_mat_;
//  cv::Mat hard_mask_; // No value pixels because of calibration

  ////////////////////////////////////////////////////////////////
  /// Storage file names.
  ////////////////////////////////////////////////////////////////
  std::string output_file_path_;
  std::string depth_file_path_;
  std::string depth_file_name_;

  ////////////////////////////////////////////////////////////////
  /// Optimization part. Related parameters.
  ////////////////////////////////////////////////////////////////
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> * pat_grid_;
  double pattern_lower_;
  double pattern_higher_;

  // Functions:
  bool LoadDatasFromFiles();
  void LoadPatternEpiInfo();
  void ConvXpro2Depth(int frm_idx, bool create_flag = true);
  void ConvDepth2Xpro(int frm_idx, bool create_flag = true);
  void FillDepthWithMask(int frm_idx);
  void SetMaskMatFromXpro(int frm_idx);

  void SetFirstFrameVertex(int frm_idx);

  void SetMaskMatFromIobs(int frm_idx);
  void PredictInitialShadeVertex(int frm_idx);
  void FillShadeMatFromVertex(int frm_idx);
  void RecoColorClass(int frm_idx);
  void GenerateIntensitySlots(int frm_idx);
  void PredictXproRange(int frm_idx);
  void PredictInitialDepthVal(int frm_idx);
  void RefineInitialDepthVal(int frm_idx);

  void SetNodeFromDepthVal(int frm_idx);
  void OptimizeDepthNode(int frm_idx);
  void SetDepthValFromNode(int frm_idx);

  void GenerateIestFromDepth(int frm_idx);
  void OptimizeShadingMat(int frm_idx);
  void OutputResult(int frm_idx);

  void ReleaseSpace(int frm_idx);

public:
  Reconstructor();
  ~Reconstructor();

  bool Init();
  bool Run();
  bool Close();
};


#endif //SLDEPTHRECONSTRUCTION_RECONSTRUCTOR_H
